
#!/bin/bash

cd /cpfs/user/chenhao/debug/
cd ./OpenRLHF_0304_vllm083/

pip3 install -e . -i  https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

pip3 install deepspeed==0.16.0

chmod -R 777 ./examples/scripts/

ln -s /cpfs/user/chenhao/debug/OpenRLHF_0304_vllm083/ /openrlhf
cd /openrlhf/examples/scripts
chmod -R 777 /openrlhf/examples/scripts

bash vllm_client_7b_local.sh


