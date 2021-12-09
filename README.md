
# DUA: Dynamic Unsupervised Adaptation 

This is the official repository for our paper: The Norm Must Go On: Dynamic Unsupervised Domain Adaptation.

Link: 

Important: Our paper is currently under-submission. We will shortly release all our code.

### Please check back later...

## Run Locally

Clone the project

```bash
  git clone https://github.com/jmiemirza/DUA.git
```

Go to the project directory

```bash
  cd DUA
```

Install dependencies

```bash
  pip install requirement.txt
```

Train a ResNet-26
```bash
  python main.py --train true --outf results/bn_adapt/
```

Test on Corruptions
```bash
   python main.py --test_c true --outf results/bn_adapt/
```

Run DUA 
```bash
  python main.py --dua true --outf results/bn_adapt/ --num_samples 5
```
Set number of *'num_samples'* argument to adapt on more samples. In the paper we always set this number to 200. Results after adaptation on all the corruptions would be saved in the directory specified for the *'outf'* argument. 

