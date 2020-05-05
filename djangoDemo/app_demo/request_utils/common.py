
def check_param_error(data, key):
    if key in data:
        return None
    response = {}
    response["status"] = -1
    response["msg"] = "missing param {}".format(str(key))
    return response

def default_post_method_error():
    response = {}
    response["status"] = -1
    response["msg"] = "must ust POST method"
    return response