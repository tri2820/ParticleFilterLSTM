"""
Target tracking dataset in Pytorch
"""
import numpy as np
import torch
from torch.utils import data


class Transfomer():
      # Todo: max-min==0 case
      def __init__(self, tf):
            self.min = torch.min(tf)
            self.max = torch.max(tf)
      def apply_(self, tf):
            tf -=self.min
            tf /=self.max-self.min
      def reverse_apply_(self, tf):
            tf *=self.max-self.min
            tf +=self.min


class TTDataset(data.Dataset):
  def __init__(self, measurement_filepath, groundtruth_filepath, transform=False, remove_dummy=False):
      'Initialization'
      self.measurement_filepath = measurement_filepath
      self.groundtruth_filepath = groundtruth_filepath

      self.measurement = torch.from_numpy(np.load(measurement_filepath))
      self.groundtruth = torch.from_numpy(np.load(groundtruth_filepath))
      
      # Check of same size
      if self.measurement.shape != self.groundtruth.shape:
            raise Exception("Mismatched size of measurement and ground truth")
      
      # Remove dummy dimension
      if remove_dummy:
            self.measurement = self.measurement.view(self.measurement.shape[0],self.measurement.shape[1],-1)
            self.groundtruth = self.groundtruth.view(self.groundtruth.shape[0],self.groundtruth.shape[1],-1)

      if transform is False or transform is None:
            self.transformer = None
      else:
            # Assign transformer else create new
            if callable(transform):
                  self.transformer = transform
            else: 
                  self.transformer = Transfomer(self.measurement)  
                  
            self.transformer.apply_(self.measurement)
            self.transformer.apply_(self.groundtruth)

  def __len__(self):
      'Denotes the total number of samples'
      return len(self.measurement)

  def __getitem__(self, idx):
      'Generates one sample of data'
      X = self.measurement[idx]
      y = self.groundtruth[idx]
      return X,y


class TTSimulationDataset(data.Dataset):
       def __init__(self, f):
            pass


if __name__ == "__main__":
      d = TTDataset(
          "data/train-data-measurements.npy",
          "data/train-data-ground_truth.npy"
          )

      shape = d.measurement.shape
      print(f"File {d.measurement_filepath} has {shape}")          
      print(f"\t{shape[0]} runs")
      print(f"\t{shape[1]} timesteps")
      print(f"\t{shape[2]} -> unclear? (dummy)")
      print(f"\t{shape[3]} tracked axises")
