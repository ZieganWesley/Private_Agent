# Installation and Run

## Install
```bash
##DIRECT INSTALL
pip install "git+https://github.com/ZieganWesley/Private_Agent.git@main"

##SOURCE INSTALL
git clone https://github.com/ZieganWesley/Private_Agent.git
cd Private_Agent
pip install -e .
```

## Run
```bash
private-agent
```

## Diagnostics
```bash
private-agent --doctor
```

## UI Check (developer only)
```bash
python -m ui.check
```
Remove this entry point before production.
