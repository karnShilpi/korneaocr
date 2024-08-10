For APK
1.Open google colb take a new Notebook
2.GO to folder->Delete all file and folder->upload your app files.
3.Go to https://towardsdatascience.com/3-ways-to-convert-python-app-into-apk-77f4c9cd55af section Google colabb ways-to-convert-python-app-into-apk-77f4c9cd55af
 Run Commands
 !pip install buildozer
 
 !pip install cython==0.29.19
 
 !sudo apt-get install -y \
    python3-pip \
    build-essential \
    git \
    python3 \
    python3-dev \
    ffmpeg \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev
    
!sudo apt-get install -y \
    libgstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good

!sudo apt-get install libffi7

!sudo apt-get install build-essential libsqlite3-dev sqlite3 bzip2 libbz2-dev zlib1g-dev libssl-dev openssl libgdbm-dev libgdbm-compat-dev liblzma-dev libreadline-dev libncursesw5-dev libffi-dev uuid-dev libffi7	
	 
!sudo apt-get install libffi-dev	
 
!buildozer init
!buildozer -v android debug


!buildozer android clean

https://docs.beeware.org/en/latest/tutorial/tutorial-5/android.html