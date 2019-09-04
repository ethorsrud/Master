import torch
import torch.nn as nn

class MultiConv1d(nn.Module):
    def __init__(self,conv_configs,in_channels,out_channels,split_in_channels=False):
        super(MultiConv1d,self).__init__()
        assert(out_channels % len(conv_configs)==0)
        self.conv_configs = conv_configs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_per_conv = in_channels
        self.out_channels_per_conv = out_channels/len(conv_configs)
        self.split_in_channels = split_in_channels

        if split_in_channels:
            assert(in_channels % len(conv_configs)==0)
            self.in_channels_per_conv = in_channels/len(conv_configs)

        self.convs = nn.ModuleList()
        for config in conv_configs:
            self.convs.append(nn.Conv1d(self.in_channels_per_conv,self.out_channels_per_conv,
                                        **config))

    def forward(self,input):
        print(input.size())
        tmp_outputs = list()
        for i,conv in enumerate(self.convs):
            tmp_input = input
            if self.split_in_channels:
                tmp_input = tmp_input[:,i*self.in_channels_per_conv:i*self.in_channels_per_conv+self.in_channels_per_conv]

            tmp_outputs.append(conv(tmp_input))

        return torch.cat(tmp_outputs,dim=1)


class MultiConv2d(nn.Module):
    def __init__(self,conv_configs,in_channels,out_channels,split_in_channels=False):
        super(MultiConv2d,self).__init__(conv_configs,in_channels,out_channels,split_in_channels)

        self.convs = nn.ModuleList()
        for config in conv_configs:
            self.convs.append(nn.Conv2d(self.in_channels_per_conv,self.out_channels_per_conv,
                                        **config))
