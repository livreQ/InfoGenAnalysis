# InfoGenAnalysis

This code repo contains the implementation of code in the ICLR 2025 paper:

**Generalization in VAE and Diffusion Models: A Unified Information-Theoretic Analysis.** by Qi Chen, Jierui Zhu, and Florian Shkurti.


## Prerequisites

To run the code, make sure that the requirements are installed.

```
pip install -r requirements.txt
```

### Run Experiment

#### VAE
1. Run the following script to get the reconstruction loss, rate, and mutual information estimation.
   ```
   cd vae
   sh run_vae.sh
   ```
2. Get the memorization score in https://github.com/alan-turing-institute/memorization.
   
  ```
  wget https://gertjanvandenburg.com/projects/memorization/results.zip # or download the file in some other way
  unzip results.zip
  touch results/*/*/*          # update modification time of the result files
  make analysis                # optionally, run ``make -n analysis`` first to see what will happen
  ```
3. Plot bounds
  ```
  python scripts/plot_bound.py 
  ```

#### Diffusion
1. Run experiments for toy data, mnist and cifar10. First configure your cluster information to run_mnist.sh and run_cifar10.sh:
   ```
   cd diffusion
   sh run_toy.sh
   sh submit_mnist.sh
   sh submit_cifar10.sh
   ``` 
2. Plot bounds
   ```
   python src/plot/plot_toy_bound.py 
   python src/plot/plot_img_bound.py 
   ```
Make sure all the paths are correct and the cluster jobs are correctly run.

### Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{qigeneralization,
  title={Generalization in VAE and Diffusion Models: A Unified Information-Theoretic Analysis},
  author={Qi, CHEN and Zhu, Jierui and Shkurti, Florian},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```


### Acknowledgments

We thank grant number DSI-PDFY3R1P11 from the Data Sciences Institute at the University of
Toronto for the support of this work.

The code in this work is heavily adapted from the following two repos:
1. For VAE, we use https://github.com/alan-turing-institute/memorization. 
2. For diffusion models, we use https://github.com/CW-Huang/sdeflow-light. 

