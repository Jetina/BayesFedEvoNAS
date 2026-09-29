class MPIMessage(object):#不同节点之间交换消息
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1 #第一次通信，用来传初始化配置
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2#将模型从服务器同步到客户端

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3#将模型从客户端发给服务器端
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4#

    MSG_ARG_KEY_OPERATION = "operation"
    MSG_ARG_KEY_TYPE = "msg_type"#消息类型
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_ARCH_PARAMS = "arch_params"
    MSG_ARG_KEY_LOCAL_TRAINING_ACC = "local_training_acc"
    MSG_ARG_KEY_LOCAL_TRAINING_LOSS = "local_training_loss"
    MSG_ARG_KEY_LOCAL_TEST_ACC = "local_test_acc"
    MSG_ARG_KEY_LOCAL_TEST_LOSS = "local_test_loss"

    MSG_OPERATION_SEND = "send"
    MSG_OPERATION_RECEIVE = "receive"
    MSG_OPERATION_BROADCAST = "broadcast"
    MSG_OPERATION_REDUCE = "reduce"

    def __init__(self):
        self._message_dict = {}#初始化一个空字典
        return

    def init(self, message_dict):
        self._message_dict = message_dict

    def add(self, key, value):
        self._message_dict[key] = value#为其添加键值

    def get(self, key):
        return self._message_dict[key]

    def get_type(self):
        return self._message_dict[MPIMessage.MSG_ARG_KEY_TYPE]

    def to_string(self):
        return self._message_dict

    def get_content(self):#获得消息内容
        print_dict = self._message_dict.copy()
        #出于隐私的考虑，因为这些参数可能包含关于正在传输的模型的敏感信息。如果看到存在于print里面，他们的值设置成无
        if MPIMessage.MSG_ARG_KEY_MODEL_PARAMS in print_dict.keys():
            print_dict[MPIMessage.MSG_ARG_KEY_MODEL_PARAMS] = None
        if MPIMessage.MSG_ARG_KEY_ARCH_PARAMS in print_dict.keys():
            print_dict[MPIMessage.MSG_ARG_KEY_ARCH_PARAMS] = None
        msg_str = self.__to_msg_type_string() + ": " + str(print_dict)#将消息类型（从_message_dict属性获得）和print_dict变量连接起来。然后，这个字符串被返回
        return msg_str
    #下面的函数是将消息类型转换成对应的字符串表示
    def __to_msg_type_string(self):
        type = self._message_dict[MPIMessage.MSG_ARG_KEY_TYPE]
        if type == MPIMessage.MSG_TYPE_S2C_INIT_CONFIG:
            type_str = "MSG_TYPE_S2C_INIT_CONFIG"
        elif type == MPIMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT:
            type_str = "MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT"
        elif type == MPIMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER:
            type_str = "MSG_TYPE_C2S_SEND_MODEL_TO_SERVER"
        elif type == MPIMessage.MSG_TYPE_C2S_SEND_STATS_TO_SERVER:
            type_str = "MSG_TYPE_C2S_SEND_STATS_TO_SERVER"
        else:
            type_str = "None"
        return type_str
