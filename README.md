[![python >3.8.8](https://img.shields.io/badge/python-3.8.8-brightgreen)](https://www.python.org/) 
[![Downloads](https://static.pepy.tech/badge/spatialid)](https://pepy.tech/project/spatialid)
#  Implementation of cell annotation method: Spatial-ID
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7340795.svg)](https://doi.org/10.5281/zenodo.7340795)           

Spatially resolved transcriptomics (SRT) provides the opportunity to investigate the gene expression profiles and the spatial context of cells in naive state. Cell type annotation is a crucial task in the spatial transcriptome analysis of cell and tissue biology. In this study, we propose Spatial-ID, a supervision-based cell typing method, for high-throughput cell-level SRT datasets that integrates transfer learning and spatial embedding. Spatial-ID effectively incorporates the existing knowledge of reference scRNA-seq datasets and the spatial information of SRT datasets.            
        
The architecture was inspired by [Spatial-ID](https://doi.org/10.1038/s41467-022-35288-0).                            
<img src="docs/source/_static/spatialID_overview.png" width="800"> 
       
### Installation      
```python
pip install SpatialID
```
        
### API        
For the API, please refer to: [https://spatialid.readthedocs.io/en/latest/index.html](https://spatialid.readthedocs.io/en/latest/index.html)

### Dependences
[![numpy-1.21.3](https://img.shields.io/badge/numpy-1.21.3-red)](https://github.com/numpy/numpy)
[![pandas-1.2.4](https://img.shields.io/badge/pandas-1.2.4-lightgrey)](https://github.com/pandas-dev/pandas)
[![scanpy-1.8.1](https://img.shields.io/badge/scanpy-1.8.1-blue)](https://github.com/theislab/scanpy)
[![torch-1.8.1](https://img.shields.io/badge/torch-1.8.1-orange)](https://github.com/pytorch/pytorch)
[![torch__geometric-1.7.2](https://img.shields.io/badge/torch__geometric-1.7.2-green)](https://github.com/pyg-team/pytorch_geometric/)

### Datasets

> - MERFISH: 280,186 cells * 254 genes, 12 samples. [https://doi.brainimagelibrary.org/doi/10.35077/g.21](https://doi.brainimagelibrary.org/doi/10.35077/g.21)
> - MERFISH-3D: 213,192 cells * 155 genes, 3 samples. [https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248)
> - Slide-seq: 207,335 cells * 27181 genes, 6 samples. [https://www.dropbox.com/s/ygzpj0d0oh67br0/Testis_Slideseq_Data.zip?dl=0](https://www.dropbox.com/s/ygzpj0d0oh67br0/Testis_Slideseq_Data.zip?dl=0)
> - NanoString: 83,621 cells * 980 genes, 20 samples. [https://nanostring.com/resources/smi-ffpe-dataset-lung9-rep1-data/](https://nanostring.com/resources/smi-ffpe-dataset-lung9-rep1-data/)
> - Stereo-Seq: Continuous slices of the mouse brain, 3 samples. [https://zenodo.org/record/7340795#.Y3xDKrZBy4S](https://zenodo.org/record/7340795#.Y3xDKrZBy4S)     
                   
### Disclaimer

This tool is for research purpose and not approved for clinical use.    
This is not an official product.     
                    
      
       
