"""Sensor-aware fusion U-Net. Copied verbatim from the training notebook (section 9.1)."""
import torch
import torch.nn as nn

STATES = {"both": (1.0, 1.0), "radar_only": (1.0, 0.0), "optical_only": (0.0, 1.0)}
ALL_STATES = list(STATES)


def _gn(c): return nn.GroupNorm(min(8, c), c)

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True),
                                   nn.Conv2d(cout, cout, 3, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class SensorAwareFusionUNet(nn.Module):
    def __init__(self, base=16, s2_channels=13, use_embed=True, use_gate=True, use_opt_head=True,
                 plain=False, emb_dim=16, drop_p=0.1):
        super().__init__()
        self.plain = plain
        self.use_embed = use_embed and not plain
        self.use_gate = use_gate and not plain
        self.use_opt_head = use_opt_head and not plain
        b = base; ch = [b, 2*b, 4*b]; self.ch = ch
        self.r_enc = nn.ModuleList([ConvBlock(2, ch[0]), ConvBlock(ch[0], ch[1]), ConvBlock(ch[1], ch[2])])
        self.o_enc = nn.ModuleList([ConvBlock(s2_channels, ch[0]), ConvBlock(ch[0], ch[1]), ConvBlock(ch[1], ch[2])])
        self.pool = nn.MaxPool2d(2); self.drop = nn.Dropout2d(drop_p)
        if self.use_embed:
            self.avail_embed = nn.Sequential(nn.Linear(2, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))
            self.film = nn.ModuleList([nn.Linear(emb_dim, 2*c) for c in ch])
            for m in self.film: nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        if self.use_gate:
            self.gates = nn.ModuleList([nn.Conv2d(2*c + (emb_dim if self.use_embed else 0), 2, 1) for c in ch])
        if self.use_opt_head:
            self.heads = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), _gn(c),
                                                      nn.ReLU(inplace=True), nn.Conv2d(c, c, 1)) for c in ch])
        n_in = 3 if self.use_opt_head else 2
        self.fuse = nn.ModuleList([ConvBlock(n_in*c, c) for c in ch])
        self.up2, self.dec2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2), ConvBlock(2*ch[1], ch[1])
        self.up1, self.dec1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2), ConvBlock(2*ch[0], ch[0])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def optical_features(self, optical):
        """Teacher features for the radar->optical head (call under no_grad on CLEAN optical)."""
        feats, x = [], optical
        for k, enc in enumerate(self.o_enc):
            x = enc(x if k == 0 else self.pool(x)); feats.append(x)
        return feats

    def _fuse(self, k, r, o, o_hat, avail, emb):
        present = avail[:, :, None, None]                          # [B,2,1,1]
        bsz, _, h, wd = r.shape
        gate_w = None
        if self.plain:
            fused_in = torch.cat([r, o], 1)                        # not told which one is missing
        else:
            if self.use_gate:
                gin = [r, o] + ([emb[:, :, None, None].expand(-1, -1, h, wd)] if emb is not None else [])
                logits = self.gates[k](torch.cat(gin, 1)).masked_fill(present == 0, -1e4)
                gate_w = torch.softmax(logits.float(), 1).to(r.dtype)
            else:                                                  # ungated: equal weight over AVAILABLE sensors
                gate_w = (present / present.sum(1, keepdim=True)).expand(-1, -1, h, wd).to(r.dtype)
            gate_w = gate_w * present.to(gate_w.dtype)             # exact zero for unavailable sensors
            parts = [r * gate_w[:, 0:1], o * gate_w[:, 1:2]]
            if self.use_opt_head: parts.append(o_hat * (1 - present[:, 1:2]).to(r.dtype))
            fused_in = torch.cat(parts, 1)
        out = self.fuse[k](fused_in)
        if emb is not None:
            gamma, beta = self.film[k](emb).chunk(2, 1)
            out = out * (1 + gamma[:, :, None, None].to(out.dtype)) + beta[:, :, None, None].to(out.dtype)
        return out, gate_w

    def forward(self, radar, optical, availability):
        if availability.ndim != 2 or availability.shape[1] != 2: raise ValueError("availability must be [B,2]")
        if bool((availability.sum(1) < 1).any()): raise ValueError("at least one sensor must be available")
        availability = availability.to(radar.dtype)
        radar = radar * availability[:, 0].view(-1, 1, 1, 1)       # hidden modality -> exactly zero input
        optical = optical * availability[:, 1].view(-1, 1, 1, 1)
        emb = self.avail_embed(availability) if self.use_embed else None
        r, o, fused, gates, o_hats = radar, optical, [], [], []
        for k in range(3):
            r = self.r_enc[k](r if k == 0 else self.pool(r))
            o = self.o_enc[k](o if k == 0 else self.pool(o))
            o_hat = self.heads[k](r) if self.use_opt_head else None   # radar features ONLY -> no leakage
            f, g = self._fuse(k, r, o, o_hat, availability, emb)
            fused.append(f); gates.append(g); o_hats.append(o_hat)
        f1, f2, f3 = fused
        f3 = self.drop(f3)
        z = self.drop(self.dec2(torch.cat([self.up2(f3), f2], 1)))
        z = self.dec1(torch.cat([self.up1(z), f1], 1))
        return self.out(z), {"gates": gates, "o_hat": o_hats}
