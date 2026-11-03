# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""octex: approximate k-nearest-neighbor search on NVIDIA GPUs (cuvs::neighbors::octex).

    import octex
    index = octex.build(xb)                # or octex.load("index_dir")
    D, I = index.search(xq, k=10)          # (distances, neighbors), nearest first

Drop-ins: `from octex import cuvs as cagra` (cuvs.neighbors.cagra calls) and `from octex import faiss` (Faiss GPU
index calls).
"""

from .index import Index, build, load

__version__ = "0.1.0"
__all__ = ["Index", "build", "load", "__version__"]
