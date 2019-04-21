# AlphaDoom

## Overview
This repository holds the final project deliverable for CSC 490, directed studies. It is an implementation of **Mastering the game of Go without human knowledge** by *David Silver et al.*

This implementation is in the 1991 3D game Doom, rather than in Go, and uses a custom next state predictor to simulate states.

## Get Started

The following information is needed in order to get this project running on your system.

### Environment

1. Create a `virtualenv` using `python3 -m virtualenv env`. Run `source env/bin/activate` to start the environment, and `deactivate` to close it.
2. Install dependencies using `pip3 install -r requirements.txt`

### Train

Run `python3 alphadoom.py` to train both the simulator and AlphaDoom

Note that this is not the intended setup, and is only being done to get around a restoring error. Ideally, one would run `python3 simulator.py` followed by `python3 alphadoom.py`.
