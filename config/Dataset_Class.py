# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# tabular datasets
class Thyroid():
    data_name = "thyroid"
    num_cls =1
class Arrhythmia():
    data_name = "arrhythmia"
    num_cls =1
class KDD():
    data_name = "kdd"
    num_cls =1
class KDDrev():
    data_name = "kddrev"
    num_cls =1

# time series datasets
class arabic_digits():
    data_name = "arabic_digits"
    num_cls =10
class characters():
    data_name = "characters"
    num_cls =20
class natops():
    data_name = "natops"
    num_cls =6
class epilepsy():
    data_name = 'epilepsy'
    num_cls = 4
class racket_sports():
    data_name = "racket_sports"
    num_cls =4

# image datasets
class fmnist():
    data_name = "fmnist"
    num_cls =10
class cifar10_feat():
    data_name = "cifar10_feat"
    num_cls =10
class galaxy():
    data_name = "galaxy"
    num_cls = 4
class galaxy_feat():
    data_name = "galaxy_feat"
    num_cls = 4