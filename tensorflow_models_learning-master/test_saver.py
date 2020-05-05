import tensorflow as tf

LEARNING_RATE_BASE = 0.1 # 设置初始学习率为0.1
LEARNING_RATE_DECAY = 0.99 # 设置学习衰减率为0.99
LEARNING_RATE_STEP = 1 # 设置喂入多少轮BATCH_SIZE之后更新一次学习率,一般设置为 总样本数/BATCH_SIZE

global_step = tf.Variable(0,trainable = False)

# 只需要这一行代码即可
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                           global_step,
                                           LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY,
                                           staircase=True)

w = tf.Variable(tf.constant(5,dtype=tf.float32))
loss = tf.square(w+1)
# 优化函数中使用前面定义好的指数衰减学习率
global_steps = tf.Variable(0,trainable = False)
learing_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_steps,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)        

optimizer = tf.train.MomentumOptimizer(learing_rate, 0.9)
train_step = optimizer.minimize(loss,global_step=global_steps)
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

with tf.Session() as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 设置训练轮数,开始训练
    for i in range(40):

        sess.run(train_step)
        # learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_steps)
        w_val = sess.run(w)
        loss_val = sess.run(loss)

        print("After {} steps:global_step is {},w is {},learning_rate is {} and loss is {}".format(i,global_step_val,w_val,1,loss_val))
