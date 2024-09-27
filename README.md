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

## Stacking

- Vertical stack (vstack) side by side
- Horizontal stack (hstack) on top of each other

## Squeezing

Remove all `1` dimensions from a tensor

## Unsqueezing

Adds a dimension with size 1 to a tensor

## Permute

Return a view of the input with dimensions permuted (swapped) in a certain way
