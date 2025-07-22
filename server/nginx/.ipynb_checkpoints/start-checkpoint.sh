

apt-get update && \
    apt-get install -y gosu && \
    rm -rf /var/lib/apt/lists/*

apt-get update && apt-get -y install sudo

apt-get install nginx -y

export MATH_VERIFY_CONF=xxx
export XVERIFY_CONF=xxx
export TIR_SANDBOX_CONF=xxx

cd /etc/nginx/conf.d
cp ${TIR_SANDBOX_CONF} ./tir_nginx.conf
cp ${MATH_VERIFY_CONF} ./math_verify_nginx.conf
cp ${XVERIFY_CONF} ./xverify_nginx.conf

nginx
nginx -t
nginx -s reload

export ADDRESS_FILE=xxxx # store machine-ip that starts nginx
rm ${ADDRESS_FILE}
export IP_ADDRESS=$(ifconfig net0 | grep 'inet ' | awk '{print $2}')
echo "IP_ADDRESS: $IP_ADDRESS"
echo "http://${IP_ADDRESS}" >> $ADDRESS_FILE

sleep 365d