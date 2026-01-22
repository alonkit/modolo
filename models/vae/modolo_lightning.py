
from torch import nn
import torch
import torch.nn.functional as F
from models.modolo_lightning import ModoloLightning
from datasets.process_chem.process_sidechains import (
    calc_tani_sim,
    get_fp,
    reconstruct_from_core_and_chains,
)
from models.vae.cvae import CyclicalAnnealer

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        # Standard CE Loss with reduction='none' so we can weight individually
        self.ce = nn.CrossEntropyLoss(weight=alpha, ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        # Calculate standard CE loss per token
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        
        # The Focal Term: (1 - pt)^gamma
        # If pt is 0.9 (easy), term is (0.1)^2 = 0.01 (Tiny gradient)
        # If pt is 0.1 (hard), term is (0.9)^2 = 0.81 (Big gradient)
        focal_loss = ((1 - pt) ** self.gamma) * -logpt


        num_tokens = torch.nonzero(targets == 2)[:,1]
        focal_loss = focal_loss * torch.log(num_tokens.float()+1).unsqueeze(1)
        
        # eoses = torch.nonzero(targets == 2)[:,1]
        # pred_eoses = torch.argmax((torch.argmax(logits,1) == 2).int(),1)        
        # early_eoses_panalty = (pred_eoses - eoses).clamp(min=0).float()
        return focal_loss[targets!=self.ignore_index].mean()

class ModoloLightningVae(ModoloLightning):
    def __init__(self, *args,**kws):
        super().__init__(*args,**kws)
        self.loss = FocalLoss(gamma=2.0, ignore_index=0)
        self.KLD_anneling = CyclicalAnnealer(n_cycles=4, max_beta=0.1, ratio=0.5)
        
    def get_loss(self,data):
        if self.handle_inactive == 'hide':
            if data.label.any(): # exist active and inactive
                # filter actives
                gs = data.to_data_list()
                gs = list(filter(lambda x: x.label.item(), gs))
                data = type(data).from_data_list(gs)
            else: # all inactive
                return torch.tensor(0.0, device=data.label.device,), {}
            
        if self.handle_inactive == 'flag':
            flag = data.label
        else:
            # flag is not interesting so..
            flag = torch.zeros_like(data.label,device=data.label.device)+1

        logits, kld_loss = self.model(
            None,
            data['ligand'].frag_tokens[:, :-1],
            data,
            (data.activity_type, flag), 
            molecule_sidechain_mask_idx=1
        )
        logits = logits.transpose(1, -1)
        tgt = data['ligand'].frag_tokens[:, 1:]
        recon_loss = self.loss(logits, tgt)
        
        kld_annealing_weight = self.KLD_anneling(
            self.trainer.global_step, 
            self.trainer.estimated_stepping_batches)
        losses = recon_loss + kld_annealing_weight * kld_loss
        return losses, {"recon": recon_loss, "kld": kld_loss, "kld_weight": kld_annealing_weight}
    
    def validation_step(self, graph, batch_idx, dataloader_idx):
        
        loss, loss_dict = self.get_loss(graph)
        self.log(f"valid_loss", loss,batch_size=len(graph), sync_dist=True)
        for k,l in loss_dict.items():
            self.log(f"valid_loss_{k}", l,batch_size=len(graph), sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)
        return {
                        "optimizer": optimizer,
                        # "lr_scheduler": {
                        #     "scheduler": sched,
                        #     "monitor": "valid_loss",
                        #     "interval": "step",
                        #     "frequency": int(self.trainer.val_check_interval * self.trainer.estimated_stepping_batches)+1,    
                        # }
                    }