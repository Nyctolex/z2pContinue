from networks import *


class PosADANet(nn.Module):
    def encode(self, shp):
        device = self.omega.device
        B, _, H, W = shp
        row = torch.arange(H).to(device) / H
        enc_row1 = torch.sin(self.omega[None, :] * row[:, None])
        enc_row2 = torch.cos(self.omega[None, :] * row[:, None])
        rows = torch.cat([enc_row1.unsqueeze(1).repeat((1, W, 1)), enc_row2.unsqueeze(1).repeat((1, W, 1))], dim=-1)

        col = torch.arange(W).to(device) / W
        enc_col1 = torch.sin(self.omega[None, :] * col[:, None])
        enc_col2 = torch.cos(self.omega[None, :] * col[:, None])
        cols = torch.cat([enc_col1.unsqueeze(0).repeat((H, 1, 1)), enc_col2.unsqueeze(0).repeat((H, 1, 1))], dim=-1)

        encoding = torch.cat([rows, cols], dim=-1)
        encoding = encoding.permute(2, 0, 1).unsqueeze(0).repeat((B, 1, 1, 1))
        return encoding

    def get_encoding(self, x):
        shp1 = x.shape
        singelton = self.positional_encoding is not None \
                    and self.positional_encoding.shape[0] == shp1[0] and self.positional_encoding.shape[2:] == shp1[2:]
        if singelton:
            return self.positional_encoding
        self.positional_encoding = self.encode(x.shape)
        return self.positional_encoding

    def __init__(self, input_channels, output_channels, n_style, bilinear=True, padding='zero', full_ada=True,
                 nfreq=20, magnitude=10, nof_layers=4, style_enc_layers=6, start_channels=64):
        super(PosADANet, self).__init__()
        factor = 2 if bilinear else 1
        self.omega = nn.Parameter(torch.rand(nfreq) * magnitude)
        self.omega.requires_grad = False
        self.positional_encoding = None
        self.full_ada = full_ada
        self.n_style = n_style

        self.style_encoder = FullyConnected(n_style, W_SIZE, layers=style_enc_layers)
        self.padding = padding
        self.input_channels = input_channels + nfreq * 4
        self.n_classes = output_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(self.input_channels, start_channels)
        self.down = nn.ModuleList([Down(start_channels*(2**i), start_channels*(2**(i+1)), padding=padding, ada=self.full_ada) for i in range(nof_layers-1)])
        self.down.append(Down(start_channels*(2**(nof_layers-1)), start_channels*(2**nof_layers) // factor, padding=padding, ada=self.full_ada))
        self.up = nn.ModuleList([Up(start_channels*(2**i), start_channels*(2**(i-1)) // factor, bilinear, ada=True, padding=padding) for i in range(nof_layers,1,-1)])
        self.up.append(Up(start_channels*2, start_channels, bilinear, ada=True, padding=padding))
        self.outc = OutConv(start_channels, output_channels, padding=padding)

    def forward(self, x, style):
        w = self.style_encoder(style) if self.full_ada else None
        encoding = self.get_encoding(x)
        x = torch.cat([x, encoding], dim=1)
        #Down pass:
        x_down = [self.inc(x)]
        for layer in self.down:
            x_down.append(layer(x_down[-1], w=w))
        #Up pass:
        x = x_down.pop(-1)
        for layer in self.up:
            x = layer(x, x_down.pop(-1), w=w)
        logits = self.outc(x)
        return logits

