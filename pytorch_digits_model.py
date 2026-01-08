from torchvision.transforms import transforms
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, RandomAffine, Normalize
from torch.autograd import Variable

class Network(nn.Module):
    def __init__(self, height, width,number_of_labels,label_encoder):
        super(Network, self).__init__()

        self.img_height = height
        self.img_width = width
        self.scaling = None  # must be set before use
        self.augment = None
        self.label_encoder = label_encoder
        self.device=None

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # Calculate the output size after all the convolutions and pooling
        def conv2d_output_size(input_size, kernel_size=5, stride=1, padding=1):
            # For non-square images, compute separately for height and width
            output_height = (input_size[0] + 2 * padding - (kernel_size - 1) - 1) // stride + 1
            output_width = (input_size[1] + 2 * padding - (kernel_size - 1) - 1) // stride + 1
            return output_height, output_width

        # Calculate size after conv1 and conv2, and after pool
        conv_size = conv2d_output_size((self.img_height, self.img_width), 5, 1, 1)
        conv_size = conv2d_output_size(conv_size, 5, 1, 1)
        pool_size = (conv_size[0] // 2, conv_size[1] // 2)

        # Calculate size after conv4 and conv5
        conv_size = conv2d_output_size(pool_size, 5, 1, 1)
        conv_size = conv2d_output_size(conv_size, 5, 1, 1)

        linear_input_features = 24 * conv_size[0] * conv_size[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(linear_input_features, number_of_labels)
        )

    def set_device(self,device):
        self.device=device

    def predict(self, input):
        input = self.preprocess(input)
        input = self.scaling(input)
        input = Variable(input.to(self.device))
        output = self.features(input)
        output = self.classifier(output)
        prob,_ = torch.max(torch.nn.functional.softmax(output, dim=1),dim=1)
        _, output = torch.max(output.data, 1)
        output = self.label_encoder.inverse_transform(output.cpu())
        return output,prob.detach().cpu().numpy()

    def predict_with_distribution(self, input):
        """Predict digits and return full softmax distribution for confidence computation."""
        input = self.preprocess(input)
        input = self.scaling(input)
        input = Variable(input.to(self.device))
        output = self.features(input)
        output = self.classifier(output)

        # Get full softmax distribution
        softmax_probs = torch.nn.functional.softmax(output, dim=1)

        # Get predicted class
        _, output = torch.max(output.data, 1)
        output = self.label_encoder.inverse_transform(output.cpu())

        return output, softmax_probs.detach().cpu().numpy()

    def forward(self, input):
        input = self.preprocess(input)
        input = self.scaling(input)
        input = Variable(input.to(self.device))
        output = self.features(input)
        output = self.classifier(output)
        return output

    def get_info(self):
        print(
            'Trained with Pytorch version 2.2.1. Expected input data type is "PIL.Image.Image" list. Example: <PIL.Image.Image image mode=RGB size=35x54 at 0x1DB74A35C50>')
        print('model trained with params: width %i, height %i, number_of_labels %i' % (self.img_width,self.img_height,len(self.label_encoder.classes_)))
        print('scaling %f, augment %f' % (self.scaling,self.augment))

    def set_scaling(self, scaling):
        self.scaling = scaling
        if self.scaling is not None:
            print('model scaling set')

    def set_augment(self, augment):
        self.augment = augment
        if self.augment is not None:
            print('using model augmentation')
        else:
            print('model augmentation disabled')

    def preprocess(self, input):
        images = []
        for image in input:

            original_width, original_height = image.size
            ratio = min(self.img_height / original_height, self.img_width / original_width)
            new_size = (int(original_height * ratio), int(original_width * ratio))
            if self.augment is not None:
                image = self.augment(image)
            image = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BICUBIC)(image)

            # Padding to achieve exactly the target size if necessary
            padding = (max(0, self.img_width - new_size[1]) // 2, max(0, self.img_height - new_size[0]) // 2)
            padding = [padding[0], padding[1], padding[0], padding[1]]

            padding[0] += self.img_width - (padding[0] * 2 + image.size[0])
            padding[1] += self.img_height - (padding[1] * 2 + image.size[1])

            image = transforms.Pad(padding, fill=0, padding_mode='edge')(image)  # left, top, right and bottom
            image = ToTensor()(image)
            images.append(image)

        return torch.stack(images)