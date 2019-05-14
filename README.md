
u_net_liver
===========
数据准备
-------
项目文件分布如下

  --project
  >main.py
  >>--data
  >>>--train
  >>>--val

模型训练
-------
python main.py train

测试模型训练
-----------
加载权重，默认保存最后一个权重

python main.py test --ckp=weights_19.pth
