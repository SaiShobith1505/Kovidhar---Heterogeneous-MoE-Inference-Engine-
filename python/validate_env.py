import subprocess
import json
import platform
import onnxruntime as ort

def get_win_info():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
        display_version = winreg.QueryValueEx(key, "DisplayVersion")[0]
        build_number = winreg.QueryValueEx(key, "CurrentBuild")[0]
        product_name = winreg.QueryValueEx(key, "ProductName")[0]
        return f"{product_name} (Build {build_number}, Version {display_version})"
    except Exception as e:
        return f"{platform.system()} {platform.release()} ({platform.version()})"

def get_npu_driver_info():
    try:
        cmd = 'Get-PnpDevice -FriendlyName "NPU Compute Accelerator Device" | Get-PnpDeviceProperty -KeyName "DEVPKEY_Device_DriverVersion" | Select-Object -ExpandProperty Data'
        res = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return "Unknown"

def main():
    providers = ort.get_available_providers()
    vitis_detected = "VitisAIExecutionProvider" in providers
    
    report = {
        "windows_version": get_win_info(),
        "onnxruntime_version": ort.__version__,
        "available_providers": providers,
        "vitis_provider_detected": vitis_detected,
        "npu_available": vitis_detected,
        "npu_driver_version": get_npu_driver_info()
    }
    
    with open("npu_environment_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("Environment validation complete. Output written to npu_environment_report.json.")
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    main()
