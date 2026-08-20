# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tunix package metadata.

Public APIs are imported from their defining submodules so importing one Tunix
utility does not eagerly load every training stack.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
  __version__ = version("google-tunix")  # match the name in pyproject.toml
except PackageNotFoundError:
  __version__ = "0.0.0.dev0"  # fallback for editable installs
