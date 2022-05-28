def convert_to_month_name(int_based_month: list) -> list:
    month_name_list = ['January', "February", 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

    try:
        return_list = [month_name_list[i - 1] for i in int_based_month]
        return return_list
    except IndexError:
        print('Month value only between 1 and 12')
    except:
        print('Failed to change to month name')
