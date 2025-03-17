# Snort Installation and Configuration Guide

## **1. Install Required Tools**
### **Download and Install Snort**
1. Download Snort for Windows from the official [Snort website](https://www.snort.org/downloads).
2. Install it to `C:\Snort`.

### **Download and Install Npcap**
1. Download [Npcap](https://nmap.org/npcap/) (WinPcap alternative).
2. Install it with default settings.

---
## **2. Verify Npcap Installation**
Run the following command to check if the `npcap` driver is running:
```sh
sc query npf
```
If it is not running, start it with:
```sh
net start npf
```

---
## **3. Verify Snort Installation**
To check if Snort is installed correctly:
```sh
snort -V
```
Expected output:
```
Version 2.9.20-WIN64 GRE (Build 82)
```

---
## **4. Set Up Environment Variables**
Manually add the Snort binary path to system environment variables:
1. Open **Command Prompt** as Administrator and run:
   ```sh
   setx /M PATH "C:\Snort\bin;%PATH%"
   ```
2. Restart the terminal and verify:
   ```sh
   snort -V
   ```

---
## **5. Verify Network Interfaces**
To find your network interface number, run:
```sh
snort -W
```
Example output:
```
   5   84:14:4D:C0:4D:29       192.168.1.22    Intel(R) Wi-Fi 6 AX201 160MHz
```
The **interface number** is `5` (Wi-Fi adapter).

---
## **6. Configure Snort Rules**
### **Set the Rules Path**
Edit `C:\Snort\etc\snort.conf` and ensure:
```sh
var RULE_PATH C:\Snort\rules
var SO_RULE_PATH ../so_rules
var PREPROC_RULE_PATH ../preproc_rules
```

### **Enable Default Rules**
Uncomment rule paths in `snort.conf`:
```sh
include $RULE_PATH/local.rules
include $RULE_PATH/app-detect.rules
```

### **Create Local Rule File**
Create `C:\Snort\rules\local.rules` and add:
```sh
alert icmp any any -> any any (msg:"ICMP detected"; sid:1000001; rev:1;)
```

---
## **7. Validate Snort Configuration**
Run Snort in test mode to check the configuration:
```sh
snort -T -c C:\Snort\etc\snort.conf
```
If successful, you will see:
```
Snort successfully validated the configuration!
Snort exiting
```

---
## **8. Run Snort for Live Traffic Capture**
### **Run in Packet Logging Mode**
```sh
snort -i 5 -l C:\Snort\log -c C:\Snort\etc\snort.conf
```

### **Run in Alert Mode**
```sh
snort -i 5 -A console -c C:\Snort\etc\snort.conf
```

---
## **9. View Logs**
Navigate to `C:\Snort\log` and open captured logs using **Wireshark**.

---
## **10. Troubleshooting**
### **No Alerts Showing?**
- Test with **ping**:
  ```sh
  ping -l 1500 8.8.8.8
  ```
- Use **Nmap**:
  ```sh
  nmap -sS -p 22,80,443 192.168.1.1
  ```

### **Run Snort in Background**
```sh
snort -i 5 -D -c C:\Snort\etc\snort.conf
```

---
### **Snort is Now Ready! 🚀**
