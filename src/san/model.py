import torch
import torch.nn as nn
from torchvision import models

# Import metadata constants from dataset
from src.san.dataset import NUM_QUESTION_TYPES, NUM_IMAGE_ORGANS

class Attention(nn.Module):
    def __init__(self, visual_dim, query_dim, attention_dim):
        super().__init__()
        self.v_proj = nn.Linear(visual_dim, attention_dim)
        self.q_proj = nn.Linear(query_dim, attention_dim)
        self.attn_proj = nn.Linear(attention_dim, 1)

    def forward(self, vi, q, return_weights=False):
        # vi: [batch, visual_dim, 7*7]
        # q: [batch, query_dim]
        
        # Project visual and query features
        vi_proj = self.v_proj(vi.transpose(1, 2)) # [batch, 49, attention_dim]
        q_proj = self.q_proj(q).unsqueeze(1)    # [batch, 1, attention_dim]
        
        # Compute attention weights
        h = torch.tanh(vi_proj + q_proj)        # [batch, 49, attention_dim]
        pi = self.attn_proj(h).squeeze(2)       # [batch, 49]
        alpha = torch.softmax(pi, dim=1)        # [batch, 49]
        
        # Compute attended visual context
        vi_attended = (vi * alpha.unsqueeze(1)).sum(dim=2) # [batch, visual_dim]
        
        if return_weights:
            return vi_attended, alpha
        return vi_attended

class SAN_RAD(nn.Module):
    def __init__(self, vocab_size, ans_vocab_size, embed_dim=300, lstm_hidden=512, 
                 attention_dim=512, qtype_embed_dim=64, organ_embed_dim=32):
        super().__init__()

        # Image encoder (extract spatial features)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # Selective unfreezing: Freeze early layers, unfreeze layer4
        for param in self.cnn[:-1].parameters():
            param.requires_grad = False
        for param in self.cnn[-1].parameters():
            param.requires_grad = True
        
        # Freeze BatchNorm for stability
        for m in self.cnn.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        
        self.cnn_dim = 2048

        # Question encoder with BIDIRECTIONAL LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)

        # NEW: Question Type Embedding
        self.qtype_embedding = nn.Embedding(NUM_QUESTION_TYPES, qtype_embed_dim)
        
        # NEW: Image Organ Embedding (for conditioning)
        self.organ_embedding = nn.Embedding(NUM_IMAGE_ORGANS, organ_embed_dim)

        # Stacked Attention layers (adjust for bidirectional LSTM + metadata embeddings)
        self.lstm_output_dim = lstm_hidden * 2  # bidirectional doubles the output
        self.combined_query_dim = self.lstm_output_dim + qtype_embed_dim + organ_embed_dim
        
        # Project combined features back to attention dimension
        self.query_proj = nn.Linear(self.combined_query_dim, self.lstm_output_dim)
        
        self.v_flat_proj = nn.Linear(self.cnn_dim, self.lstm_output_dim)
        self.attn1 = Attention(self.cnn_dim, self.lstm_output_dim, attention_dim)
        self.attn2 = Attention(self.cnn_dim, self.lstm_output_dim, attention_dim)

        # Answer decoder
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, ans_vocab_size),
        )
        
    def forward(self, images, questions, q_type_idx=None, organ_idx=None):
        # Image spatial features: [batch, 2048, 7, 7]
        vi = self.cnn(images)
        batch_size = vi.size(0)
        vi = vi.view(batch_size, self.cnn_dim, -1) # [batch, 2048, 49]

        # Question features: [batch, lstm_hidden*2] for bidirectional
        emb = self.embedding(questions)
        _, (hidden, _) = self.lstm(emb)
        q = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch, lstm_hidden*2]

        # NEW: Add question type and organ embeddings if provided
        if q_type_idx is not None and organ_idx is not None:
            qtype_embed = self.qtype_embedding(q_type_idx)  # [batch, qtype_embed_dim]
            organ_embed = self.organ_embedding(organ_idx)    # [batch, organ_embed_dim]
            
            # Concatenate all features
            q_combined = torch.cat([q, qtype_embed, organ_embed], dim=1)  # [batch, combined_dim]
            
            # Project back to original dimension for attention
            q = self.query_proj(q_combined)  # [batch, lstm_output_dim]

        # Layer 1 Attention
        vi_attended1 = self.attn1(vi, q)
        vi_attended1_proj = self.v_flat_proj(vi_attended1)
        u1 = torch.tanh(vi_attended1_proj + q)

        # Layer 2 Attention
        vi_attended2 = self.attn2(vi, u1)
        vi_attended2_proj = self.v_flat_proj(vi_attended2)
        u2 = torch.tanh(vi_attended2_proj + u1)

        return self.classifier(u2)
    
    def forward_with_attention(self, images, questions, q_type_idx=None, organ_idx=None):
        """
        Forward pass that returns both prediction and attention weights for visualization.
        
        Returns:
            output: Model predictions [batch, num_classes]
            attention_weights: Attention weights from layer 2 [batch, 49]
        """
        # Image spatial features
        vi = self.cnn(images)
        batch_size = vi.size(0)
        vi = vi.view(batch_size, self.cnn_dim, -1)

        # Question features
        emb = self.embedding(questions)
        _, (hidden, _) = self.lstm(emb)
        q = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # Add metadata embeddings if provided
        if q_type_idx is not None and organ_idx is not None:
            qtype_embed = self.qtype_embedding(q_type_idx)
            organ_embed = self.organ_embedding(organ_idx)
            q_combined = torch.cat([q, qtype_embed, organ_embed], dim=1)
            q = self.query_proj(q_combined)

        # Layer 1 Attention
        vi_attended1 = self.attn1(vi, q)
        vi_attended1_proj = self.v_flat_proj(vi_attended1)
        u1 = torch.tanh(vi_attended1_proj + q)

        # Layer 2 Attention (with weights for visualization)
        vi_attended2, attn_weights = self.attn2(vi, u1, return_weights=True)
        vi_attended2_proj = self.v_flat_proj(vi_attended2)
        u2 = torch.tanh(vi_attended2_proj + u1)

        output = self.classifier(u2)
        return output, attn_weights

