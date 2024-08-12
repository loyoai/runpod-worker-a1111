import os
import time
import requests
import traceback
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from requests.adapters import HTTPAdapter, Retry
from huggingface_hub import HfApi
from schemas.input import INPUT_SCHEMA
from schemas.api import API_SCHEMA
from schemas.img2img import IMG2IMG_SCHEMA
from schemas.txt2img import TXT2IMG_SCHEMA
from schemas.interrogate import INTERROGATE_SCHEMA
from schemas.sync import SYNC_SCHEMA
from schemas.download import DOWNLOAD_SCHEMA

BASE_URI = 'http://127.0.0.1:3000'
TIMEOUT = 600
POST_RETRIES = 3

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
#                               Utility Functions                              #
# ---------------------------------------------------------------------------- #

def wait_for_service(url):
    retries = 0

    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                logger.info('Service not ready yet. Retrying... humm')
        except Exception as err:
            logger.error(f'Error: {err}')

        time.sleep(0.2)


def send_get_request(endpoint):
    return session.get(
        url=f'{BASE_URI}/{endpoint}',
        timeout=TIMEOUT
    )


def send_post_request(endpoint, payload, job_id, retry=0):
    response = session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )

    # Retry the post request in case the model has not completed loading yet
    if response.status_code == 404:
        if retry < POST_RETRIES:
            retry += 1
            logger.warn(f'Received HTTP 404 from endpoint: {endpoint}, Retrying: {retry}', job_id)
            time.sleep(0.2)
            send_post_request(endpoint, payload, job_id, retry)

    return response


def validate_input(job):
    return validate(job['input'], INPUT_SCHEMA)


def validate_api(job):
    api = job['input']['api']
    api['endpoint'] = api['endpoint'].lstrip('/')

    return validate(api, API_SCHEMA)


def validate_payload(job):
    method = job['input']['api']['method']
    endpoint = job['input']['api']['endpoint']
    payload = job['input']['payload']
    validated_input = payload

    if endpoint == 'v1/sync':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, SYNC_SCHEMA)
    elif endpoint == 'v1/download':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, DOWNLOAD_SCHEMA)
    elif endpoint == 'sdapi/v1/txt2img':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, TXT2IMG_SCHEMA)
    elif endpoint == 'sdapi/v1/img2img':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, IMG2IMG_SCHEMA)
    elif endpoint == 'sdapi/v1/interrogate' and method == 'POST':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, INTERROGATE_SCHEMA)

    return endpoint, job['input']['api']['method'], validated_input


def download(job):
    source_url = job['input']['payload']['source_url']
    download_path = job['input']['payload']['download_path']
    process_id = os.getpid()
    temp_path = f"{download_path}.{process_id}"

    # Download the file and save it as a temporary file
    with requests.get(source_url, stream=True) as r:
        r.raise_for_status()
        with open(temp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Rename the temporary file to the actual file name
    os.rename(temp_path, download_path)
    logger.info(f'{source_url} successfully downloaded to {download_path}', job['id'])

    return {
        'msg': 'Download successful',
        'source_url': source_url,
        'download_path': download_path
    }


def sync(job):
    repo_id = job['input']['payload']['repo_id']
    sync_path = job['input']['payload']['sync_path']
    hf_token = job['input']['payload']['hf_token']

    api = HfApi()

    models = api.list_repo_files(
        repo_id=repo_id,
        token=hf_token
    )

    synced_count = 0
    synced_files = []

    for model in models:
        folder = os.path.dirname(model)
        dest_path = f'{sync_path}/{model}'

        if folder and not os.path.exists(dest_path):
            logger.info(f'Syncing {model} to {dest_path}', job['id'])

            uri = api.hf_hub_download(
                token=hf_token,
                repo_id=repo_id,
                filename=model,
                local_dir=sync_path,
                local_dir_use_symlinks=False
            )

            if uri:
                synced_count += 1
                synced_files.append(dest_path)

    return {
        'synced_count': synced_count,
        'synced_files': synced_files
    }


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #

def send_warmup_request():
    warmup_payload = {
        "input": {
            "api": {
                "method": "POST",
                "endpoint": "sdapi/v1/img2img",
            },
            "payload": {
                "alwayson_scripts": {
                    "Soft Inpainting": {
                        "args": [True, 1, 0.5, 4, 0, 0.5, 2]
                    },
                    "controlnet": {
                        "args": [
                            {
                                "advanced_weighting": None,
                                "batch_images": "",
                                "control_mode": "Balanced",
                                "enabled": True,
                                "guidance_end": 0.7,
                                "guidance_start": 0,
                                "hr_option": "Both",
                                "image": [
                                        {"image":"https://iomzilgmkxhrgzkzmntc.supabase.co/storage/v1/object/public/generated/ef5dcd1f-a3c2-4f04-985c-06badf173b2d.png"},
                                        {"image":"https://iomzilgmkxhrgzkzmntc.supabase.co/storage/v1/object/public/generated/4f24942a-bccb-48e3-9b0b-74ae0cd20989.png"},
                                        {"image":"https://iomzilgmkxhrgzkzmntc.supabase.co/storage/v1/object/public/generated/71fcfa3c-be6e-4751-b107-9fa4b1b2e4af.png"}
                                    ],
                                "inpaint_crop_input_image": True,
                                "input_mode": "merge",
                                "ipadapter_input": None,
                                "is_ui": True,
                                "loopback": False,
                                "low_vram": False,
                                "model": "ip-adapter-faceid-plusv2_sdxl [187cb962]",
                                "module": "ip-adapter_face_id_plus",
                                "output_dir": "",
                                "pixel_perfect": True,
                                "processor_res": 512,
                                "resize_mode": "Crop and Resize",
                                "save_detected_map": True,
                                "threshold_a": 0.5,
                                "threshold_b": 0.5,
                                "weight": 1.89
                            },
                        ],
                    },
                },
                "batch_size": 1,
                "cfg_scale": 4.5,
                "scheduler": "automatic",
                "mask": "https://iomzilgmkxhrgzkzmntc.supabase.co/storage/v1/object/public/generated/assets/mask2.png",
                "comments": {},
                "denoising_strength": 0.8,
                "disable_extra_networks": False,
                "do_not_save_grid": False,
                "do_not_save_samples": False,
                "height": 1024,
                "image_cfg_scale": 3.5,
                "init_images": ["https://iomzilgmkxhrgzkzmntc.supabase.co/storage/v1/object/public/generated/assets/inpaint_jpg.jpg"],
                "initial_noise_multiplier": 1,
                "inpaint_full_res_padding": 32,
                "inpainting_fill": 1,
                "inpainting_mask_invert": 0,
                "mask_blur": 4,
                "mask_blur_x": 4,
                "mask_blur_y": 4,
                "n_iter": 1,
                "negative_prompt": "angry, sad, bad mouth",
                "override_settings": {},
                "override_settings_restore_afterwards": True,
                "prompt": "women, long hair, <lora:ip-adapter-faceid-plusv2_sdxl_lora:0.8>",
                "resize_mode": 0,
                "restore_faces": False,
                "s_churn": 0,
                "s_min_uncond": 0,
                "s_noise": 1,
                "s_tmax": None,
                "s_tmin": 0,
                "sampler_name": "Euler a",
                "script_args": [],
                "script_name": None,
                "seed": -1,
                "seed_resize_from_h": -1,
                "seed_resize_from_w": -1,
                "steps": 30,
                "styles": [],
                "subseed": -1,
                "subseed_strength": 0,
                "tiling": False,
                "width": 1024,
                "override_settings": {
                    "CLIP_stop_at_last_layers": 2,
                },
                "override_settings_restore_afterwards": True,
            },
        },
    }

    def send_single_request(request_number):
        logger.info(f"Sending warm-up request #{request_number}")
        response = send_post_request("sdapi/v1/img2img", warmup_payload["input"]["payload"], f"warmup_{request_number}")
        
        if response.status_code == 200:
            logger.info(f"Warm-up request #{request_number} completed successfully")
        else:
            logger.error(f"Warm-up request #{request_number} failed. Status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")

    # Send first warm-up request
    send_single_request(1)

    # Send second warm-up request
    send_single_request(2)



def handler(job):
    validated_input = validate_input(job)

    if 'errors' in validated_input:
        return {
            'error': '\n'.join(validated_input['errors'])
        }

    validated_api = validate_api(job)

    if 'errors' in validated_api:
        return {
            'error': '\n'.join(validated_api['errors'])
        }

    endpoint, method, validated_payload = validate_payload(job)

    if 'errors' in validated_payload:
        return {
            'error': '\n'.join(validated_payload['errors'])
        }

    if 'validated_input' in validated_payload:
        payload = validated_payload['validated_input']
    else:
        payload = validated_payload

    try:
        logger.info(f'Sending {method} request to: /{endpoint}', job['id'])

        if endpoint == 'v1/download':
            return download(job)
        elif endpoint == 'v1/sync':
            return sync(job)
        elif method == 'GET':
            response = send_get_request(endpoint)
        elif method == 'POST':
            response = send_post_request(endpoint, payload, job['id'])

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f'HTTP Status code: {response.status_code}', job['id'])
            logger.error(f'Response: {response.json()}', job['id'])

            return {
                'error': f'A1111 status code: {response.status_code}',
                'output': response.json(),
                'refresh_worker': True
            }
    except Exception as e:
        logger.error(f'An exception was raised: {e}')

        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }


if __name__ == "__main__":
    wait_for_service(f'{BASE_URI}/sdapi/v1/sd-models')
    send_warmup_request()
    logger.info('A1111 Stable Diffusion API is ready')
    logger.info('Starting RunPod Serverless...')
    logger.info('Test - Rami')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
