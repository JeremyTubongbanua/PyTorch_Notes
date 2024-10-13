# PyTorch_Notes

Personal notes for PyTorch

## Tensor Types

### Scalar

0-dimensional tensor

```python
scalar = torch.tensor(7)
```

### Vector

1-dimensional tensor

```python
vector = torch.tensor([1, 2, 3])
```

### Matrix

2-dimensional tensor

```python
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
```

### Tensor

n-dimensional tensor

```python
tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
```

## Random Tensor

```python
tensor = tensor.rand(2, 3) # generates a random tensor with size 2x3
```

## Zero Tensor

```python
tensor = tensor.zeroes(3, 3) # generates a 3x3 tensor of zeroes
```

## Ones Tensor

```python
tensor = tensor.ones(3, 3) # generates a 3x3 tensor of ones
```

## Zero Tensor Like

```python
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor_like = torch.zeros_like(tensor) # generates a tensor of zeroes with the same size as tensor
```

## arange

arange is similar to the range function in Python

```python
tensor = torch.arange(0, 10, 2) # generates a tensor with values from 0 to 10 with step 2
```

## Tensor Element Multiplication

```python
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
result = tensor1 * tensor2 # element multiplication
# result = tensor1.mul(tensor2) # another way to do element multiplication
```

## Tensor Matrix Multiplication

```python
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(tensor1, tensor2) # matrix multiplication
result = tensor1.matmul(tensor2) # another way to do matrix multiplication
result = tensor1 @ tensor2 # another way to do matrix multiplication
result = tensor1.mm(tensor2) # another way to do matrix multiplication
result = torch.mm(tensor1, tensor2) # another way to do matrix multiplication
```

## Rules for Tensor Multiplication

1. inner dimensions must match
2. the resulting matrix has the shape of the outer dimensions

## Tensor Transpose

```python
tensor = torch.tensor([[1, 2], [3, 4]])
result = tensor.T # transpose
# result becomes [[1, 3], [2, 4]]
```

## Min and Max

```python
tensor = torch.arange(0, 100, 20)
result = tensor.min() # returns 0
```

```python
tensor = torch.arange(0, 100, 20)
result = tensor.max() # returns 80
```

## argmin and argmax

```python
tensor = torch.tensor([1, 2, 3, 4, 5])
result = tensor.argmax() # returns 4
```

```python
tensor = torch.tensor([1, 2, 3, 4, 5])
result = tensor.argmin() # returns 0
```

## Reshaping

Reshapes an input tensor to a defined shape

For example,

```python
# this is a 2d matrix
tensor = torch.tensor([[1, 2, 3]])
# the shape is (1, 3)

# now I want it to be shape (3)
tensor = tensor.reshape(3)
tensor # tensor([1, 2, 3])
```

## View

Return a view of an input tensro of certain shape but keep the same memory as the original tensor

```python
x = torch.tensor([[1, 2, 3]])
z = x.view(3)
z[0] = 1000

print(x) # tensor([[1000, 2, 3]])
print(z) # tensor([1000, 2, 3])
```

## Stacking

- Vertical stack (vstack) side by side
- Horizontal stack (hstack) on top of each other

```python
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
torch.stack([tensor1, tensor2], dim=0) # vertical stack
# output is tensor([[1, 2, 3], [4, 5, 6]])

torch.stack([tensor1, tensor2], dim=1) # horizontal stack
# output is tensor([[1, 4], [2, 5], [3, 6]])
```

## Squeezing

Remove all `1` dimensions from a tensor

```python
# Squeeze
## removes all single dimensions from a target tensor

tensor1 = torch.tensor([[1, 2, 3]])
print(f"Unsqueezed tensor1: {tensor1}")
print(f"Unsqueezed tensor1 shape: {tensor1.shape}")
print(f"Squeezed tensor1: {tensor1.squeeze()}")
print(f"Squeezed tensor1 shape: {tensor1.squeeze().shape}")
```

Output

```
Unsqueezed tensor1: tensor([[1, 2, 3]])
Unsqueezed tensor1 shape: torch.Size([1, 3])
Squeezed tensor1: tensor([1, 2, 3])
Squeezed tensor1 shape: torch.Size([3])
```

## Unsqueezing

Adds a dimension with size 1 to a tensor

```python
tensor1 = torch.tensor([1, 2, 3])
print(f"Tensor: {tensor1} - shape = {tensor1.shape}")
unsqueezed_tensor1 = tensor1.unsqueeze(dim=0)
print(f"Tensor unsqueezed at dim=0: {unsqueezed_tensor1} - shape = {unsqueezed_tensor1.shape}")
unsqueezed_tensor2 = unsqueezed_tensor1.unsqueeze(dim=2)
print(f"Tensor unsqueezed at dim=0: {unsqueezed_tensor2} - shape = {unsqueezed_tensor2.shape}")
```

```
Tensor: tensor([1, 2, 3]) - shape = torch.Size([3])
Tensor unsqueezed at dim=0: tensor([[1, 2, 3]]) - shape = torch.Size([1, 3])
Tensor unsqueezed at dim=0: tensor([[[1],
         [2],
         [3]]]) - shape = torch.Size([1, 3, 1])
```

## Permute

Return a view of the input with dimensions permuted (swapped) in a certain way

Also shares the same memory as the original tensor

```python
# torch.pernute rearranges dimensions
tensor = torch.arange(0, 100)
print(tensor.shape)
new_tensor = tensor.reshape(2, 10, 5)
print(new_tensor.shape)
permuted = new_tensor.permute(2, 0, 1)
print(permuted.shape)
```

```
torch.Size([100])
torch.Size([2, 10, 5])
torch.Size([5, 2, 10])
```

## Indexing

```python
tensor = torch.tensor([[8, 7, 1],
        [8, 5, 1],
        [3, 7, 2]])

tensor[:, 1]
# tensor([7, 5, 7])
```

## Controled RNG

```python
torch.manual_seed(42)
tensor_1 = torch.rand(2, 2)
print(tensor_1)

torch.manual_seed(42)
tensor_2 = torch.rand(2,2)
print(tensor_2)

print(tensor_1 == tensor_2)
```

## Using CUDA

Setting up device

```python
import torch
if torch.cuda.is_available():
    device = "cuda"
else"
    device = "cpu"
```

Moving it to a device

```python
tensor = torch.tensor([1,2,3])
tensor_on_gpu = tensor.to(device)
tensor_on_gpu
```

```python
tensor = torch.tensor([1, 2, 3], device=device)
```

Bring it back to CPU

```python
tensor_on_cpu = tensor_on_gpu.to("cpu")
```
