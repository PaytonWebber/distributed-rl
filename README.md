# Distributed AlphaZero

This repository contains a distributed deep reinforcement learning (DDRL) system based on the AlphaZero algorithm. Our project evaluates the effectiveness of scaling actor-learner pipelines using consumer-grade hardware. We propose a distributed architecture where multiple self-play actors generate experience asynchronously and push it to a central replay buffer, which a learner then samples from to update a shared neural network
