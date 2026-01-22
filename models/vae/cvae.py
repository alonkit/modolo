import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, d_model, d_latent):
        super().__init__()
        
        # --- 1. POSTERIOR ENCODER (Training Only) ---
        # Takes BOTH Anchor and Target Fragment Info
        # We need an encoder to process the SMILES fragment into a vector first
        self.fragment_encoder = nn.Linear(d_model, d_model) 
        
        # Predicts z based on (Anchor + Fragment)
        # self.mu_net = nn.Linear(d_model * 2, d_latent)
        # self.logvar_net = nn.Linear(d_model * 2, d_latent)
        hidden_dim = d_model * 2
        dropout = 0.1
        self.mu_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_latent)
        )
        
        self.logvar_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_latent)
        )
        
        # --- 2. DECODER COMPONENTS ---
        self.fusion = nn.Linear(d_model + d_latent, d_model)
        self.d_latent = d_latent
        
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, anchor, fragment_embedding=None, training=True):
        """
        anchor: [Batch, d_model]
        fragment_embedding: [Batch, d_model] (Summary of the ground truth fragment)
        """
        
        if training:
            # --- TRAINING PATH ---
            # 1. Encode the fragment (Ground Truth)
            frag_feat = self.fragment_encoder(fragment_embedding)
            
            # 2. Concatenate Anchor + Fragment
            combined_input = torch.cat([anchor, frag_feat], dim=-1)
            
            # 3. Predict z specific to THIS fragment
            mu = self.mu_net(combined_input)
            logvar = self.logvar_net(combined_input)
            z = self.reparameterize(mu, logvar)
            
            # Calculate KLD Loss against a Standard Normal Prior N(0,1)
            # This forces the learned z's to stay close to 0,1 so we can sample them later
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kld_loss = kld_loss / anchor.shape[0]  # Average over batch
        else:
            # --- INFERENCE PATH ---
            # We DON'T have the fragment.
            # We ignore mu_net and logvar_net entirely.
            # We sample z directly from N(0,1)
            
            z = torch.randn(anchor.shape[0], self.d_latent).to(anchor.device)
            kld_loss = 0 # No KLD during inference

        # --- DECODING (Shared) ---
        # Combine Anchor + z (The specific instruction)
        decoder_input = torch.cat([anchor, z], dim=-1)
        decoder_input = self.fusion(decoder_input)
        
        return decoder_input, kld_loss
    
class CyclicalAnnealer:
    def __init__(self, n_cycles=4, max_beta=0.02, ratio=0.5):
        """
        n_cycles: How many times to restart the annealing (usually 4)
        max_beta: The maximum weight for KLD (the target 'reining in' force)
        ratio: What % of the cycle is spent increasing beta (0.5 = 50% rising, 50% flat)
        """
        self.n_cycles = n_cycles
        self.max_beta = max_beta
        self.ratio = ratio
        self.step_counter = 0

    def __call__(self, step, total_steps):
        # Calculate where we are in the current cycle (0.0 to 1.0)
        self.period = total_steps / self.n_cycles
        
        cycle_progress = (step % self.period) / self.period
        
        if cycle_progress < self.ratio:
            scale = cycle_progress / self.ratio
        else:
            scale = 1.0
            
        return scale * self.max_beta
