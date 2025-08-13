import torch

path = r"C:\Users\Jang Sehyuk\Like_lion\Fish_detector\database.pt"
data = torch.load(path)
unique_names = set(data['labels'])
print(len(unique_names))
print(unique_names)

