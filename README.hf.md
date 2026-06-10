---
title: Nexus-AI-Gateway
emoji: 🪐
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Nexus AI Gateway Space

This space hosts the FastAPI inference, simulation, and GPT intelligence server for Project Nexus.

## Local Development
To run this container locally:
```bash
docker build -t nexus-gateway -f Dockerfile.hf .
docker run -p 8080:7860 nexus-gateway
```
