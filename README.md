# BidNet  
### Compute-Aware Neural Architecture via Differentiable Bidding

---

## Overview

This project explores how neural networks can adapt how much computation they use depending on the input.

Most models use a fixed amount of compute for every example. In this project I built a modular network made up of several experts and a routing mechanism that decides how much each expert should contribute. The idea is to let the model allocate compute dynamically rather than treating every input the same.

The routing is learned and differentiable, so the whole system can be trained end to end.

---

## Motivation

Following my final stage interview at the company Graphcore, I really liked the work that they were doing in optimising chips for AI. Following some conversations with engineers there, I realised a lot of modern models are quite inefficient in the sense that they apply the same amount of computation to easy and difficult inputs. That does not seem ideal.

I wanted to explore whether a relatively simple system could learn to reduce compute where it is not needed while still maintaining reasonable accuracy.

This led to the idea of experts competing for compute through a bidding mechanism.

---

## Approach

The model consists of:

- a shared convolutional backbone  
- multiple expert networks  
- a routing module that assigns weights to each expert  

Each expert produces a score for how useful it would be for the current input. These scores are turned into weights using a softmax, so experts effectively compete with each other.

The final prediction is a weighted combination of the expert outputs.

---

## Compute-aware training

To encourage the model to use less compute, I added a penalty term to the loss:

L = classification loss + lambda * compute

Here compute is approximated using the total expert activation. Lambda controls how much the model is penalised for using more experts.

I also implemented a simple feedback rule that adjusts lambda during training based on how far the current compute is from a target value. This is loosely inspired by Lagrangian methods for constrained optimisation.

---

## Routing variants

I implemented three routing modes so I could compare behaviour:

- dense where all experts are always active  
- top-k where a fixed number of experts are selected  
- softmax where experts compete and share compute  

The softmax version is the main one used in BidNet.

---

## Results

On CIFAR-10 the model behaves as expected.

Top-k routing uses a fixed amount of compute per input and reaches around mid 70 percent accuracy after a few epochs.

The softmax routing model achieves similar or better accuracy while using less effective compute on average. With the compute penalty turned on the model becomes more selective and reduces expert usage further.

---

## Sparsity

One of the main things I looked at was how many experts are actually active.

As training progresses and the compute penalty increases, the model starts to concentrate most of the probability mass on a small number of experts. This leads to higher sparsity even though the routing is still fully differentiable.

---

## Expert specialisation

I also logged expert usage per class.

Different classes tend to rely on different experts. For example some classes consistently use one expert much more than others, while other classes split more evenly. This behaviour emerges without any explicit supervision.

---

## FLOPs

I did not measure exact FLOPs using a profiler. Instead I used a simple proxy based on expert activation. This is enough to compare relative compute across different routing strategies.

---

## Running the project

Install dependencies:

pip install -r requirements.txt

Train the model:

python scripts/train.py --config configs/dense.yaml

---

## Configuration

Key settings:

- routing_type can be dense, topk or softmax  
- lambda_compute controls the strength of the compute penalty  
- target_compute sets the desired compute level  
- lambda_lr controls how quickly lambda is updated  

---

## Reflections

The main takeaway is that even a fairly simple setup can learn to adjust how much compute it uses. The model does not explicitly know which inputs are easy or hard, but it still learns to route them differently.

The current implementation has some limitations. In particular the softmax routing keeps total compute roughly constant, so the model expresses efficiency through sparsity rather than reducing total activation. This could be improved with a different gating mechanism.

---

## Future work

Some natural extensions would be:

- using a routing mechanism that allows total compute to vary more freely  
- measuring real FLOPs rather than using a proxy  
- scaling to larger models and datasets  

---
