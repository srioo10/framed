"""Lab attacks for the live Veto test (PREREG §9 LIVE). Run only on the isolated lab (bigp-iot behind the Veto gateway,
targets = our own Pi 192.168.77.1 and sensor), never with an internet uplink on the lab side. Every run appends one
ground-truth line {kind, src, dst, t_start, t_end, params} (epoch ms) to --log, which labels the flows afterwards.
Raw-packet kinds need cap_net_raw on ~/.venvs/bigp-attack/bin/python3 (and on nmap), set once by the user.

  synscan     nmap SYN scan of 1000 ports (nmap -sS --privileged, -S src)
  udpscan     UDP probes to N ports, rate-capped
  synflood    TCP SYNs to dport at --pps (capped at 200) for --secs; random source ports
  sshbrute    SSH logins with wrong passwords (paramiko), one TCP session per try
  mqttbrute   MQTT CONNECTs with wrong passwords, one TCP session per try
  mqttflood   valid MQTT session publishing at --pps for --secs (lab client credentials)
  spoof       forged flows naming --victim as source (The