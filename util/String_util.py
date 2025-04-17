def convert_string_bool_to_bool(value):
    if value is not None:
        return value.lower() == 'true'

def convert_empty_string_to_none(value):
    if value =="":
        return None