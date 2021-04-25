import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNet(nn.Module):

    def __init__(self, observation_shape, num_actions, layers=(2,3,4,5,6)):
        super(ResNet, self).__init__()
        
        self.block = BasicBlock
        self.num_actions = num_actions
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        #self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(observation_shape[0], self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, layers[0])
        self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(self.block, 1024, layers[4], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * self.block.expansion, 256)

        self.total_cut_layers = 10

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # FC output size + last reward.
        core_output_size = self.fc.out_features + 1

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        return tuple()
        
    #def forward(self, x, start_cut_point = 0, stop_cut_point = 12):
    def forward(self, inputs, core_state, cut_layer = 10, LearnerInferenceMode=False, T=0, B=0, reward=[]):
        netStr = [
            "self.conv1(x)",
            #"self.bn1(x)",
            "self.relu(x)",
            "self.maxpool1(x)",
            "self.layer1(x)",
            "self.layer2(x)",
            "self.maxpool2(x)",
            "self.layer3(x)",
            "self.layer4(x)",
            "self.layer5(x)",
            "self.avgpool(x)"
            ]

        
        if(not LearnerInferenceMode):
            x = inputs["frame"]
            T, B, *_ = x.shape
            x = torch.flatten(x, 0, 1)  # Merge time and batch.
            x = x.float() / 255.0

            layer_index = 0

            while(layer_index < cut_layer):
                x = eval(netStr[layer_index],{"self":self,"torch":torch,"x":x})
                layer_index = layer_index + 1

            if(cut_layer < self.total_cut_layers):
                return x, T, B

            clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)

        else:
            x = inputs
            layer_index = cut_layer
            
            while(layer_index < self.total_cut_layers):
                x = eval(netStr[layer_index],{"self":self,"torch":torch,"x":x})
                layer_index = layer_index + 1

            clipped_reward = torch.clamp(reward, -1, 1).view(T * B, 1)

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        core_input = torch.cat([x, clipped_reward], dim=-1)

        core_output = core_input

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        #norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                #norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))#, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))#,norm_layer=norm_layer))

        return nn.Sequential(*layers)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        #if norm_layer is None:
        #    norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

if __name__ == '__main__':
    model = ResNet((4,84,84),num_actions=6)
    print(model)
