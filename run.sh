./build.sh
if [ $? -eq 0 ]; then
	./test.sh
else
	echo "build.sh failed, skipping tests."
	exit 1
fi