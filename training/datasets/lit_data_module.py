# Copyright (C) 2025 Rafael Luque Tejada
# Author: Rafael Luque Tejada <lukemaster.master@gmail.com>
#
# This file is part of Generación de Música Personalizada a través de Modelos Generativos Adversariales.
#
# Euterpe as a part of the project Generación de Música Personalizada a través de Modelos Generativos Adversariales is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Generación de Música Personalizada a través de Modelos Generativos Adversariales is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

class LitDataModule(pl.LightningDataModule):
    def __init__(self, dataset_cls=None, dataset_kwargs=None,train_dataset=None,val_dataset=None, val_split=0.2, batch_size=1, num_workers=0):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs or {}
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        if self.train_dataset is not None and self.val_dataset is None:
            total_len = len(self.train_dataset)
            train_len = int((1 - self.val_split) * total_len)
            val_len = total_len - train_len
            self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_len, val_len])
            return

        if self.dataset_cls is None:
            raise ValueError("It's necessary to provide 'dataset_cls' or an initializced dataset.")

        full_dataset = self.dataset_cls(**self.dataset_kwargs)
        train_size = int((1 - self.val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)