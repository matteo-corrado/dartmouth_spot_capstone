# Remote Access with Tailscale

Tailscale provides secure SSH access to the Jetson from anywhere — on campus, at home, or on the go. It creates a private mesh VPN so you can connect without port forwarding or firewall rules.

## Why Tailscale?

The Jetson sits on Spot's WiFi network (`192.168.80.x`), which is a private network with no internet access. Without Tailscale, you can only reach the Jetson when physically connected to the same network. Tailscale gives the Jetson a stable IP on a private overlay network (the "tailnet") that works from anywhere with internet.

```
┌────────────┐     Tailscale      ┌──────────────────┐     Spot WiFi     ┌────────────┐
│  Your       │ ◄──────────────► │  Jetson AGX Orin  │ ◄──────────────► │  Spot Robot │
│  Laptop     │   (100.x.x.x)    │  (192.168.80.x)   │   (192.168.80.3)  │             │
└────────────┘                   └──────────────────┘                   └────────────┘
```

The Jetson connects to both networks simultaneously: Spot's WiFi for robot control, and Tailscale (over Ethernet or a second WiFi adapter) for remote access.

## Install Tailscale on the Jetson

```bash
# Install Tailscale (one-line installer)
curl -fsSL https://tailscale.com/install.sh | sh

# Start Tailscale and authenticate
sudo tailscale up

# Follow the printed URL to log in with your Tailscale account
# The Jetson will appear in your tailnet as "jetson-agx-orin" (or the hostname)
```

After authentication, verify:

```bash
tailscale ip -4
# Prints something like: 100.64.0.5
```

This `100.x.x.x` address is your Jetson's Tailscale IP. It stays the same even if the Jetson's local IP changes.

!!! note "Tailnet account"
    Ask a team member which Tailscale account (tailnet) to use. Everyone on the team should join the same tailnet to access the Jetson.

## Enable Tailscale SSH

Tailscale SSH lets you connect without managing SSH keys:

```bash
sudo tailscale up --ssh
```

This enables Tailscale's built-in SSH server. You can now SSH into the Jetson from any machine on the same tailnet.

## Connect from Your Laptop

### 1. Install Tailscale on your machine

Download from [tailscale.com/download](https://tailscale.com/download) and log in to the same tailnet.

### 2. SSH into the Jetson

```bash
# Using Tailscale IP
ssh spotdog@100.x.x.x

# Or using the Tailscale hostname (check your admin console)
ssh spotdog@jetson-agx-orin
```

If you enabled Tailscale SSH (`--ssh`), no password or key is needed — Tailscale handles authentication.

### 3. Verify you can reach Spot through the Jetson

Once SSHed in, confirm the Jetson can still talk to Spot:

```bash
ping 192.168.80.3
```

## Access the Web Panel Remotely

The web panel (`scripts/web_panel.py`) binds to `0.0.0.0:8080`, so it is accessible via the Jetson's Tailscale IP:

```
http://100.x.x.x:8080
```

Open this URL on your phone or laptop (as long as you are on the same tailnet) to control E-Stop and the voice pipeline remotely.

!!! warning "No authentication on the web panel"
    The web panel has no password protection. Anyone on your tailnet can trigger E-Stop or start/stop the voice pipeline. This is acceptable for a private tailnet but keep the tailnet membership restricted.

## Ensure Tailscale Starts on Boot

Tailscale installs as a systemd service and starts automatically. Verify:

```bash
sudo systemctl is-enabled tailscaled
# Should print: enabled

# If not:
sudo systemctl enable tailscaled
```

After a Jetson reboot, Tailscale reconnects automatically within 10-20 seconds. You do not need to run `tailscale up` again.

## Internet Access on the Jetson

The Jetson needs internet for:

- Pulling Ollama models (`ollama pull`)
- Downloading YOLO weights (first run)
- Tailscale connectivity
- `apt` updates and package installs

Spot's WiFi does **not** provide internet. You need a second network connection:

| Method | Setup |
|--------|-------|
| **Ethernet** (recommended) | Plug an Ethernet cable from a campus network port into the Jetson |
| **USB WiFi adapter** | Add a second WiFi interface for campus/home WiFi |
| **USB tethering** | Share your phone's internet via USB |

Tailscale handles routing automatically — traffic to `192.168.80.x` goes over Spot's WiFi, everything else goes over the internet-connected interface.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `tailscale up` hangs | Check internet connectivity: `ping 8.8.8.8`. Tailscale needs internet to authenticate. |
| Can't SSH after reboot | Wait 20 seconds for Tailscale to reconnect, then retry. Check `sudo systemctl status tailscaled`. |
| Tailscale IP changed | Run `tailscale ip -4` on the Jetson. IPs should be stable, but check if the node was removed/re-added. |
| Web panel unreachable via Tailscale | Verify `web_panel.py` is running. Try `curl http://localhost:8080` on the Jetson first. |
| Spot unreachable from Jetson | Tailscale shouldn't affect local routing. Check `ip route` and confirm `192.168.80.0/24` still routes over the Spot WiFi interface. |
