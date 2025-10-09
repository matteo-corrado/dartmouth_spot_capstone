from bosdyn.client import create_standard_sdk
def main():
    sdk = create_standard_sdk("my-spot-app")
    print("SDK imported, app skeleton OK")
if __name__ == "__main__":
    main()
