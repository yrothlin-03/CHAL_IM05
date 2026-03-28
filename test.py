import os, sys, time, json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models
from PIL import Image
import warnings; warnings.filterwarnings('ignore')

BASE_PATH = Path("/home/infres/yrothlin-24/CHAL_IM05")
TRAIN_CSV = BASE_PATH / "data/IMA205-challenge" / "train_metadata.csv"
TRAIN_DIR = BASE_PATH / "data/IMA205-challenge" / "train"
TEST_DIR = BASE_PATH / "data/IMA205-challenge" / "test"
CHECKPOINT_DIR = BASE_PATH / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ['BA','BL','BNE','EO','LY','MMY','MO','MY','PC','PLY','PMY','SNE','VLY']
CLASS_TO_IDX = {cls:idx for idx,cls in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx:cls for cls,idx in CLASS_TO_IDX.items()}

CONFIG = {'batch_size':32,'num_epochs':80,'learning_rate':1e-4,'weight_decay':1e-5,
'patience':25,'min_delta':0.001,'image_size':300,'num_classes':13,
'device':'cuda' if torch.cuda.is_available() else 'cpu','num_workers':4,'seed':42,
'clip_grad_norm':1.0,'label_smoothing':0.05,'use_mixup':False,'warmup_epochs':10}

torch.manual_seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])
if torch.cuda.is_available(): torch.cuda.manual_seed_all(CONFIG['seed'])

class BloodCellDatasetFromCSV(Dataset):
    def __init__(self,csv_path=None,img_dir=None,split='train',transform=None,return_id=False):
        self.img_dir = Path(img_dir) if img_dir else (TRAIN_DIR if split!='test' else TEST_DIR)
        self.transform,self.split,self.return_id = transform,split,return_id
        self.image_ids,self.labels = [],[]
        if split=='test':
            for img_path in sorted(self.img_dir.glob('*.png')): self.image_ids.append(img_path.name)
            print(f"✅ Test: {len(self.image_ids)} images")
        else:
            if csv_path is None: csv_path=TRAIN_CSV
            df=pd.read_csv(csv_path); available=set(p.name for p in self.img_dir.glob('*.png'))
            df=df[df['ID'].isin(available)]; self.image_ids=df['ID'].tolist()
            self.labels=[CLASS_TO_IDX[label] for label in df['label']]
            if split=='train':
                counts=np.bincount(self.labels,minlength=CONFIG['num_classes'])
                print(f"✅ Train: {len(self.image_ids)} images | Distribution:")
                for cls,count in zip(CLASS_NAMES,counts):
                    pct=count/len(self.labels)*100; marker="⚠️ " if count<50 else ""
                    print(f"   {marker}{cls:4s}: {count:5d} ({pct:5.1f}%)")
                self.class_weights=1.0/(counts+1e-5); self.class_weights/=self.class_weights.sum()*CONFIG['num_classes']
    def __len__(self): return len(self.image_ids)
    def __getitem__(self,idx):
        img_id=self.image_ids[idx]; img_path=self.img_dir/img_id
        try: image=Image.open(img_path).convert('RGB')
        except: image=Image.new('RGB',(CONFIG['image_size'],CONFIG['image_size']))
        if self.transform: image=self.transform(image)
        if self.split=='test': return (image,img_id) if self.return_id else image
        return image,self.labels[idx],img_id

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((CONFIG['image_size'],CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(p=0.3),transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=8),transforms.ColorJitter(brightness=0.1,contrast=0.15),
        transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize((CONFIG['image_size'],CONFIG['image_size'])),
        transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

class EfficientNetB3Custom(nn.Module):
    def __init__(self,num_classes=13,pretrained=True):
        super().__init__()
        if pretrained:
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
            base=models.efficientnet_b3(weights=weights)
            self.features,self.avgpool=base.features,base.avgpool
            self.classifier=nn.Sequential(nn.Dropout(0.4),nn.Linear(base.classifier[1].in_features,1024),
            nn.SiLU(),nn.Dropout(0.3),nn.Linear(1024,num_classes))
        else: raise ValueError("Pré-entraîné requis")
    def forward(self,x):
        x=self.features(x); x=self.avgpool(x); x=torch.flatten(x,1); return self.classifier(x)

class TrainerStable:
    def __init__(self,model,train_loader,val_loader,class_weights=None,checkpoint_path=None):
        self.model,self.train_loader,self.val_loader=model.to(CONFIG['device']),train_loader,val_loader
        self.checkpoint_path=checkpoint_path or CHECKPOINT_DIR/f"efficientnet_b3_stable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        if class_weights is not None: class_weights=torch.FloatTensor(class_weights).to(CONFIG['device'])
        self.criterion=nn.CrossEntropyLoss(weight=class_weights,label_smoothing=CONFIG['label_smoothing'])
        self.optimizer=optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
        self.scheduler=optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=CONFIG['num_epochs']-CONFIG['warmup_epochs'],eta_min=1e-6)
        self.best_val_f1,self.patience_counter,self.history,self.start_epoch=0.0,0,defaultdict(list),0
        self.last_lr=CONFIG['learning_rate']
        if checkpoint_path and checkpoint_path.exists():
            try:
                ckpt=torch.load(checkpoint_path,map_location=CONFIG['device'])
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt: self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                self.best_val_f1,self.history,self.start_epoch=ckpt['best_val_f1'],defaultdict(list,ckpt['history']),ckpt['epoch']+1
                print(f"✅ Reprise depuis epoch {self.start_epoch} (meilleur F1: {self.best_val_f1:.4f})")
            except Exception as e: print(f"⚠️ Erreur chargement checkpoint: {e}")
    def _save_checkpoint(self,epoch,val_f1):
        if val_f1>self.best_val_f1+CONFIG['min_delta']:
            self.best_val_f1,self.patience_counter=val_f1,0
            ckpt={'epoch':epoch,'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),'scheduler_state_dict':self.scheduler.state_dict(),
            'best_val_f1':self.best_val_f1,'history':dict(self.history),'config':CONFIG,'class_names':CLASS_NAMES}
            torch.save(ckpt,self.checkpoint_path); print(f"   💾 Meilleur modèle sauvegardé (F1={val_f1:.4f})")
            return True
        else: self.patience_counter+=1; return False
    def _print_progress(self,epoch,batch_idx,total,loss,mode='Train'):
        pct=(batch_idx+1)/total*100; bar='█'*int(pct/5)+'░'*(20-int(pct/5))
        sys.stdout.write(f"\r{mode} Epoch {epoch+1:3d} [{bar}] {pct:5.1f}% Loss: {loss:.4f}"); sys.stdout.flush()
    def train_epoch(self,epoch):
        self.model.train(); total_loss=0
        if epoch<CONFIG['warmup_epochs']:
            lr=CONFIG['learning_rate']*(epoch+1)/CONFIG['warmup_epochs']
            for pg in self.optimizer.param_groups: pg['lr']=lr
        for i,(images,labels,_) in enumerate(self.train_loader):
            images,labels=images.to(CONFIG['device']),labels.to(CONFIG['device'])
            self.optimizer.zero_grad(); outputs=self.model(images)
            loss=self.criterion(outputs,labels); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),CONFIG['clip_grad_norm'])
            self.optimizer.step(); total_loss+=loss.item()
            if (i+1)%10==0 or i==len(self.train_loader)-1: self._print_progress(epoch,i,len(self.train_loader),loss.item())
        print(); return total_loss/len(self.train_loader)
    @torch.no_grad()
    def validate(self,epoch):
        self.model.eval(); all_preds,all_labels,total_loss=[],[],0
        for images,labels,_ in self.val_loader:
            images,labels=images.to(CONFIG['device']),labels.to(CONFIG['device'])
            outputs=self.model(images); loss=self.criterion(outputs,labels); total_loss+=loss.item()
            all_preds.extend(torch.argmax(outputs,dim=1).cpu().numpy()); all_labels.extend(labels.cpu().numpy())
        from sklearn.metrics import f1_score
        f1_macro=f1_score(all_labels,all_preds,average='macro')
        f1_weighted=f1_score(all_labels,all_preds,average='weighted')
        counts=np.bincount(all_preds,minlength=CONFIG['num_classes'])
        if np.max(counts)/len(all_preds)>0.85: print(f"   ⚠️  ALERTE: Une seule classe dominante ({np.max(counts)/len(all_preds):.1%})")
        lr=self.optimizer.param_groups[0]['lr']
        if lr!=self.last_lr and epoch>=CONFIG['warmup_epochs']: print(f"   🔁 LR: {self.last_lr:.2e} → {lr:.2e}"); self.last_lr=lr
        print(f"   ✅ Val Loss: {total_loss/len(self.val_loader):.4f} | F1 Macro: {f1_macro:.4f} | F1 Weighted: {f1_weighted:.4f}")
        return total_loss/len(self.val_loader),f1_macro,f1_weighted
    def train(self):
        print("\n"+"="*70); print("🚀 ENTRAÎNEMENT STABILISÉ AVEC EFFICIENTNET-B3"); print("="*70)
        print(f"Device: {CONFIG['device']} | LR: {CONFIG['learning_rate']} | Batch: {CONFIG['batch_size']}")
        print(f"Label Smoothing: {CONFIG['label_smoothing']} | MixUp: DÉSACTIVÉ | Warmup: {CONFIG['warmup_epochs']} epochs")
        print("="*70+"\n")
        for epoch in range(self.start_epoch,CONFIG['num_epochs']):
            print(f"\n{'='*70}\n🔄 EPOCH {epoch+1}/{CONFIG['num_epochs']}\n{'='*70}")
            train_loss=self.train_epoch(epoch); print(f"   ✅ Train Loss: {train_loss:.4f}")
            val_loss,val_f1_macro,val_f1_weighted=self.validate(epoch)
            self.history['train_loss'].append(train_loss); self.history['val_loss'].append(val_loss)
            self.history['val_f1_macro'].append(val_f1_macro); self.history['val_f1_weighted'].append(val_f1_weighted)
            improved=self._save_checkpoint(epoch,val_f1_macro)
            if val_f1_weighted<0.5 and epoch>15: print(f"\n⚠️  WARNING: F1 Weighted faible ({val_f1_weighted:.4f}) après 15 epochs")
            if self.patience_counter>=CONFIG['patience']: print(f"\n⏰ Early stopping après {epoch+1} epochs"); break
        print("\n"+"="*70); print(f"✅ ENTRAÎNEMENT TERMINÉ - Meilleur F1 Macro: {self.best_val_f1:.4f}"); print("="*70)
        return self.history
    @torch.no_grad()
    def predict_test(self,test_loader):
        print("\n"+"="*70); print("🔮 PRÉDICTION SUR LE TEST SET"); print("="*70)
        self.model.eval(); all_preds,all_ids=[],[]; total=len(test_loader)
        for i,(images,img_ids) in enumerate(test_loader):
            images=img_ids=images.to(CONFIG['device']),img_ids
            preds=torch.argmax(self.model(images),dim=1); all_preds.extend(preds.cpu().numpy()); all_ids.extend(img_ids)
            if (i+1)%10==0 or i==total-1:
                pct=(i+1)/total*100; bar='█'*int(pct/5)+'░'*(20-int(pct/5))
                sys.stdout.write(f"\rPrédiction [{bar}] {pct:5.1f}%"); sys.stdout.flush()
        print("\n✅ Prédiction terminée")
        labels=[IDX_TO_CLASS[p] for p in all_preds]; return pd.DataFrame({'ID':all_ids,'label':labels})

def main(resume_from=None):
    print("\n"+"="*70); print("🔬 IMA2026 CHALLENGE - EFFICIENTNET-B3 STABILISÉ"); print("="*70)
    for path,name in [(TRAIN_CSV,"train_metadata.csv"),(TRAIN_DIR,"train/"),(TEST_DIR,"test/")]:
        if not path.exists(): raise FileNotFoundError(f"❌ {name} manquant: {path}")
    print("\n📥 Chargement des données...")
    full=BloodCellDatasetFromCSV(csv_path=TRAIN_CSV,img_dir=TRAIN_DIR,split='train',transform=get_train_transforms())
    from sklearn.model_selection import train_test_split
    idx=np.arange(len(full)); train_idx,val_idx=train_test_split(idx,test_size=0.15,stratify=full.labels,random_state=CONFIG['seed'])
    train_loader=DataLoader(full,batch_size=CONFIG['batch_size'],sampler=SubsetRandomSampler(train_idx),
    num_workers=CONFIG['num_workers'],pin_memory=True)
    val=BloodCellDatasetFromCSV(csv_path=TRAIN_CSV,img_dir=TRAIN_DIR,split='val',transform=get_val_test_transforms())
    val_loader=DataLoader(val,batch_size=CONFIG['batch_size']*2,sampler=SubsetRandomSampler(val_idx),
    num_workers=CONFIG['num_workers'],pin_memory=True)
    test=BloodCellDatasetFromCSV(img_dir=TEST_DIR,split='test',transform=get_val_test_transforms(),return_id=True)
    test_loader=DataLoader(test,batch_size=CONFIG['batch_size']*2,shuffle=False,
    num_workers=CONFIG['num_workers'],pin_memory=True)
    print(f"\n✅ Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test)}")
    print("\n🧠 Création d'EfficientNet-B3 (pré-entraîné)...")
    model=EfficientNetB3Custom(num_classes=CONFIG['num_classes'],pretrained=True)
    print(f"✅ Modèle sur {CONFIG['device']} | Taille: {CONFIG['image_size']}x{CONFIG['image_size']}")
    print("\n⚡ Démarrage de l'entraînement STABILISÉ...")
    trainer=TrainerStable(model,train_loader,val_loader,getattr(full,'class_weights',None),
    Path(resume_from) if resume_from else None)
    history=trainer.train()
    hist_path=CHECKPOINT_DIR/f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(hist_path,'w') as f: json.dump({k:[float(v) for v in vs] if vs else [] for k,vs in history.items()},f,indent=2)
    print(f"\n📊 Historique sauvegardé: {hist_path.name}")
    print("\n📤 Génération de la soumission...")
    if trainer.checkpoint_path.exists():
        ckpt=torch.load(trainer.checkpoint_path,map_location=CONFIG['device'])
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ Meilleur modèle chargé (F1={ckpt['best_val_f1']:.4f})")
    submission=trainer.predict_test(test_loader)
    sub_path=BASE_PATH/f"submission_efficientnet_b3_stable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    submission.to_csv(sub_path,index=False)
    print(f"\n✅ Soumission créée: {sub_path.name}")
    print(f"\n📊 Distribution des prédictions:\n{submission['label'].value_counts().sort_index().to_string()}")
    print("\n"+"="*70); print("✨ TERMINÉ - Score attendu: F1 Macro 0.70-0.78"); print("="*70)
    return submission

if __name__=="__main__":
    import argparse; parser=argparse.ArgumentParser()
    parser.add_argument('--resume',type=str,default=None,help='Checkpoint pour reprendre')
    args=parser.parse_args(); main(resume_from=args.resume)
