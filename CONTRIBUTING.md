### Contributor policy

I have a very little time and need to be picky about contributions, so please make sure you contribute code that is:
- well thought
- covered with unit tests
- you understand everything what you wrote
- as concise as possible - I can't handle too big PRs, sorry!

Use LLM at your discretion, but provide exhaustive explanation of what you built and why. Write it by yourself to show that you really understand

I can understand if that sounds too picky, but since I build this project after hours, I need to cut any additional noise. Sorry and thanks for understanding!

### Build from source (only for development)

1. Clone this repo
2. Build Dawn: `./scripts/build-dawn.sh` (or set `DAWN_PREFIX` to your Dawn installation)
3. Build: `./build.sh`

### C++ unit tests

0. Remember to rebuild your code before testing - `./build.sh`
1. `chmod +x build-ctests.sh run-ctests.sh`
2. Update `build-ctests.sh` with your paths
3. `rm -rf build/ctests`
4. `./build-ctests.sh`
5. `./run-ctests.sh`

### C++ benchmarks

0. Remember to rebuild your code before testing - `./build.sh` and optionally log in to your wandb.ai account
1. `chmod +x build-benchmark.sh run-benchmark.sh`
2. Update `build-benchmark.sh` with your paths
3. `rm -rf build/benchmarks`
4. `./build-benchmark.sh`
5. `./run-benchmark.sh`

### Python unit tests

0. Remember to rebuild your code before testing - `./build.sh`
1. `pytest tests` to run all tests. `pytest tests/ops/test_cos.py` to run a chosen test file, like here we test cosinus
