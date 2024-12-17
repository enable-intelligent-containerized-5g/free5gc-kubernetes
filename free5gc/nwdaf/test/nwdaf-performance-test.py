import requests, sys, time, base64
from datetime import datetime, timedelta
import concurrent.futures

def make_request(url, method, headers=None, params=None, data=None, json=None):
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            data=data,
            json=json
        )
        
        print(f"Status Code: {response.status_code}")
        return response
    except requests.RequestException as e:
        print(f"Error in the request: {e}")
        return None
    
def date_formated(date):
    return date.strftime('%Y-%m-%dT%H:%M:%S.') + f'{date.microsecond // 1000:03d}Z'

def load_dataset_in_base64(path):
    with open(path, 'rb') as file:
        file_content = file.read()
    encoded_content = base64.b64encode(file_content).decode('utf-8')
    
    return f"data:text/csv;base64,{encoded_content}"

def main():
    
    if len(sys.argv) != 3:
        print("Usage: python3 nwdaf-performance-test.py <test-type {s -> statistics, p -> predictios, t -> training} > <number-of-test>")
        sys.exit(1)
    # Get params  
    test_type = sys.argv[1]
    # s -> statistics, p -> predictios, t -> training 
    test_number = int(sys.argv[2])
    
    # Common config
    headers = {"Content-Type": "application/json"}
    params = None
    data = None
    tp = 1
    metric = "NF_LOAD" 
    current_time = datetime.now()
    current_time_utc5 = current_time + timedelta(hours=5)
    # AnLF
    anlf_url = "http://127.0.0.1:30080/"
    nfs = ["AUSF"]
    nf_ids = ["386eb6ba-d314-4a34-9f98-90d194718c7e"]
    is_nf = True
    # MTLF
    mtlf_url = "http://127.0.0.1:30081/"
    using_pcm = True
    data_source = "./"
    
    if is_nf:
        content_type = "nfTypes"
        content_type_data = nfs
    else:
        content_type = "nfInstanceIds"
        content_type_data = nf_ids
    
    
    if test_type == "s":
        start_time = current_time_utc5 + timedelta(minutes=-tp)
        
        url = f"{anlf_url}nnwdaf-analyticsinfo/v1/analyticsinfo/request"
        method = "POST"
        json_data = {
            "endTime": date_formated(current_time_utc5),
            "eventId": metric,
            content_type: content_type_data,
            "startTime": date_formated(start_time)
        }
        
    elif test_type == "p":
        end_time = current_time_utc5 + timedelta(minutes=tp)
        
        url = f"{anlf_url}nnwdaf-analyticsinfo/v1/analyticsinfo/request"
        method = "POST"
        json_data = {
            "endTime": date_formated(end_time),
            "eventId": metric,
            content_type: content_type_data,
            "startTime": date_formated(current_time_utc5)
        }
        
    elif test_type == "t":
        start_time = current_time_utc5 + timedelta(hours=-5)
        
        if tp == 1:
            dataset = 'dataset_NF_LOAD_AMF_60s_1733807760_1733915795.csv'
        elif tp == 2:
            dataset = 'dataset_NF_LOAD_AMF_120s_1733807760_1733915725.csv'
        else:
            sys.exit(f"Undefined tartget period: {tp}")
        
        if using_pcm :
            dataset_name = None
            dataset_base64 = None
        else:
            dataset_name = dataset
            dataset_base64 = load_dataset_in_base64(f"{data_source}{dataset}")
        
        url = f"{mtlf_url}nnwdaf-mlmodeltraining/v1/mlmodeltraining/request"
        method = "POST"
        json_data = {
            "eventId": metric,
            "file": { 
                "data": dataset_base64,
                "name": dataset_name,
            },
            "newDataset": True,
            "nfType": nfs[0],
            "startTime": date_formated(start_time),
            "targetPeriod": tp*60
        }
        
    else:
        sys.exit(f"Undefined test type: {test_type}")
        

    # Run the request
    total_time = 0
    for i in range(test_number):
        print(f"Request #{i+1}")
        
        start_time = time.time()
        response = make_request(url, method, headers=headers, params=params, data=data, json=json_data)
        end_time = time.time()
        
        request_time = round(end_time - start_time, 2)
        total_time += request_time
        if response:
            print(f"Request #{i+1} successful ({request_time} seconds\n")
        else:
            print(f"Request #{i+1} failed.\n") 
    
    average_time = total_time/test_number
    print(f"Average time: {round(average_time, 2)} seconds")
    print(f"Request rate: {round(test_number/total_time, 2)} requests/s")

if __name__ == "__main__":
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     resultados = list(executor.map(tarea, trabajos))
    main()