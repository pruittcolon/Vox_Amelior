
Skip to content
Navigation Menu

    Pricing

Sign in
Sign up
JohnRThomas /
EvenDemoApp
Public
forked from even-realities/EvenDemoApp

Code
Pull requests
Actions
Projects
Wiki
Security

    Insights

Even Realities G1 BLE Protocol
John Thomas edited this page last month · 8 revisions

The Even Realities G1 operates with two BLE radios, one for the left side and one for the right. Some commands need to be sent individually to both sides and some are only sent to one side. Each radio operates as a stream of packets using the Nordic BLE UART.

TODO: Describe the NRF UART BLE UUIDs TODO: Explain Global Sequence
Operations
Display Image Full Screen

TODO
Display Image On Dashboard

TODO
Display Text Full Screen

TODO
Display Text On Dashboard

TODO
AI Dictation and Response

TODO
Create and Display Quick Note

TODO
Navigation Full Screen

TODO
Notifications

TODO
Record Audio

TODO
User Defined Action on Double Tap

TODO
Scroll Text Full Screen

The official app uses this for the teleprompter feature, but it can scroll any text such as a news feed or an ebook.
Display Calendar on Dashboard

TODO
Display Graph on Dashboard

The official app uses this for Stocks, but it can be used for anything, for example heart rate over time or blood glucose levels.
Messages

These are messages the glasses will send to the app unprompted when an event happens or they want to put the status in.
Status (0x22)

These messages come when certain events happen on the glasses. See: Dashboard Set (0x06) for the Dashboard and Pane modes.

Sent when the user looks up.
Header 	Size 	?? 	?? 	Event? 	Num Unread 	Low Power 	Dashboard Mode 	Pane Mode 	Pane Page Number
22 	0A 	00 	00 	01 	00 ~ FF 	00 / 01 	00 ~ 02 	00 ~ 05 	01 ~ 04

Sent when the user taps the right side while looking up.
Header 	Size 	?? 	?? 	Event? 	Dashboard Mode 	Pane Mode 	Pane Page Number
22 	08 	00 	00 	02 	00 ~ 02 	00 ~ 05 	01 ~ 04
Audio (0xF1)

A continuous Audio Stream sent when the user looks up and Head Up Microphone is enabled or the Microphone is manually enabled. See: Activate Microphone on Head Lift Settings (0x08) and Microphone Set (0x0E). The Audio Sequence is different from the Global Sequence, and will reset each time the microphone is enabled or disabled.
Header 	Audio Sequence 	Audio Data (LC3)
F1 	00 ~ FF 	XX XX ...
Debug (0xF4)

The glasses send debug messages when debugging mode is enabled.The message is a null terminated ASCII string. See Debug Mode Set (0x23 0x6C)
Header 	Message
F4 	Null Terminated ASCII Data.\0
Event (0xF5)
Header 	Sub 	Payload 	Description
F5 	00 		TouchPad Double Tap
F5 	01 		TouchPad Single Tap
F5 	02 		Head Up
F5 	03 		Head Down
F5 	04 		TouchPad Triple Tap
F5 	05 		TouchPad Triple Tap
F5 	06 		Worn
F5 	07 		Not worn / Not in case
F5 	08 		In Case / Lid open
F5 	09 	00 / 01 	Glasses side charging
F5 	0A 	00 ~ 64 	Glasses side battery level
F5 	0B 		In case / Lid closed / Case plugged in
F5 	0C 		
F5 	0D 		
F5 	0E 	00 / 01 	Case charging
F5 	0F 	00 ~ 64 	Case battery level
F5 	10 		
F5 	11 		BLE Paired Success?
F5 	12 		Right TouchPad pressed, held and released
F5 	17 		Left TouchPad pressed and held
F5 	18 		Left TouchPad pressed Released
F5 	19 		
F5 	1A 		
F5 	1B 		
F5 	1C 		
F5 	1D 		
F5 	1E 		Open Dashboard (double tap)
F5 	1F 		Close Dashboard (double tap)
F5 	20 		Double tap either side when set to translate or transcribe
Commands

Commands are messages the App can send to the glasses. The glasses may optionally respond.
Generic Command Response

Many commands will simply respond with a generic success/failure message. The header will repeat the command id back, and one additional byte which will be either success or failure. This is recorded here and referenced by all commands that operate this way to avoid duplicating the same response for all such messages.
Command 	Subcommand 	Description
XX 	C9 	Success
XX 	CA 	Failure
XX 	CB 	Continue data
Anti Shake Get (0x2A)
This is not used by the stock app. TODO
Bitmap Show (0x16)

Shows a previously uploaded file as a bitmap on the screen. The image file must be uploaded before it can be displayed. See: File Upload (0x15)
Command 	File ID 	CRC
16 	0D 0E 	

The crc value is calculated using Crc32Xz big endian, combined with the bmp picture storage address and picture data.
Bitmap Hide (0x18)

Clears the screen of bitmaps. Also clears text? TODO
Brightness Set (0x01)

Adjust the brightness level or enable/disable auto brightness. Send to Right side. Response is generic.
Command 	Brightness 	Auto
01 	00 ~ 2A 	00 / 01
Brightness Get (0x29)

Fetches the current screen brightness. Send to Right side.
Command
29
Response
Header 	?? 	Brightness Value 	Auto brightness Enabled
29 	65 	00 ~ 2A 	00 / 01
Dashboard Set (0x06)

Send settings to control the dashboard. The length includes full packet length, including the command, length, pad, sequence and subcommand payload. Subcommands are listed below with their payloads.
Command 	Length 	Pad 	Sequence
06 	XX 	00 	00 ~ FF
Subcommand Payload
XX XX XX XX ...
Response
Command 	Request Length 	Pad 	Sequence 	Subcommand 	Chunk Count 	Pad 	Chunk 	Pad 	Success
06 	XX 	00 	00-FF 	XX 	01 ~ FF 	00 	01 ~ FF 	00 	00 / 01
Time and Weather Set (0x01)
Command 	Length 	Pad 	Sequence
06 	16 	00 	00 ~ FF
Subcommand 	Epoch Time 32bit (s) 	Epoch Time 64bit (ms) 	Weather Icon ID 	Temp C 	C/F 	24H/12H 	??
01 	XX XX XX XX 	XX XX XX XX XX XX XX XX 	01-10 	XX 	00/01 	00/01 	00
Weather Set (0x02)

TODO: This is just a guess
Command 	Length 	Pad 	Sequence
06 	08 	00 	00 ~ FF
Subcommand 	Weather Icon ID 	Temp C 	C/F
02 	01-10 	XX 	00/01
Weather Icon IDs
Icon ID 	Description 		Icon ID 	Description
00 	None 		09 	Snow
01 	Night 		0A 	Mist
02 	Clouds 		0B 	Fog
03 	Drizzle 		0C 	Sand
04 	Heavy Drizzle 		0D 	Squalls
05 	Rain 		0E 	Tornado
06 	Heavy Rain 		0F 	Freezing
07 	Thunder 		10 	Sunny
08 	Thunder Storm 		11+ 	Error
Pane Mode Set (0x06)

Set the dashboard mode, currently has three modes: minimal, dual, full
Command 	Length 	Pad 	Sequence
06 	07 	00 	00 ~ FF
Subcommand 	Mode ID 	Secondary Pane ID
06 	00 ~ 02 	00 ~ 05
Dashboard Mode IDs
Mode ID 	Description
00 	Full
01 	Dual
02 	Minimal
Secondary Pane IDs

Only respected on Full or Dual Mode
Secondary Pane ID 	Description
00 	Notes
01 	Stock (graph)
02 	News
03 	Calendar
04 	Map
05 + 	Empty
Pane Calendar Set (0x03)

Sets the data to be shown on the calendar pane of the Dashboard.
Command 	Length 	Pad 	Sequence
06 	XX 	00 	00 ~ FF
Subcommand 	Chunk Count 	Pad 	Chunk 	Pad 	?? 	Number of Events 	Events...
03 	01 ~ FF 	00 	01 ~ FF 	00 	01 03 03 	XX 	XX XX ...

Chunk Indexes at 1 not 0 for this command.

Each event entry will be composed of three strings that will take up two lines in the calendar. The official app uses Text 01 as the Title, Text 02 as the time, and Text 03 as the location.
Text (01) 	
Text (02) 	Text (03)

The event entries are sent in a list of entries each with three strings. Each string can have a length of 0 to indicate the field is blank.
Text 	Len 	Value(ASCII) 	Text 	Len 	Value(ASCII) 	Text 	Len 	Value(ASCII)
01 	01~FF 	XX XX ... 	02 	01 ~ FF 	XX XX ... 	03 	01~FF 	XX XX ...
Pane Stock Set (0x04)

TODO
Command 	Length 	Pad 	Sequence
06 	XX 	00 	00 ~ FF
Subcommand 	Chunk Count 	Pad 	Chunk 	Pad 			
04 	01 ~ FF 	00 	01 ~ FF 	00 			
Pane News Set (0x05)

TODO
Command 	Length 	Pad 	Sequence
06 	XX 	00 	00 ~ FF
Subcommand 	Chunk Count 	Pad 	Chunk 	Pad 			
05 	01 ~ FF 	00 	01 ~ FF 	00 			
Pane Map Set (0x07)

TODO
Command 	Length 	Pad 	Sequence
06 	XX 	00 	00 ~ FF
Subcommand 	Chunk Count 	Pad 	Chunk 	Pad 			
07 	01 ~ FF 	00 	01 ~ FF 	00 			
Send Arrow Data

TODO
Dashboard Calendar Next Up Set (0x58)

Sets the calendar event displayed below the clock then the dashboard is in Full Mode.

TODO
Dashboard Quick Note Set (0x1E)
TODO
File Upload (0x15)

TODO
File Upload Complete (0x20)

Also used to upgrade a font, it seems. TODO
Hardware Set (0x26)
Response
Header 	Packet size 	Pad 	Seq 	SubCommand 	Success
26 	06 	00 	00-FF 	XX 	C9/CA
Subcommands
Subcommand ID 	Description
01 	Set something to 1
02 	Set Display Height and Depth
03 	give mic_transm_sem
04 	ble set lum gear
05 	Set Double Tap Action
06 	ble set lum coeffic
07 	Enable/Disable Long Press Action
08 	Set Activate Mic on Head Up
Height and Depth (0x02)

Control the display’s height and depth. Must be called twice, first with the preview bit set to 1, and then a few seconds later with the bit set 0. The glasses will stay on permanently until the preview=0 command is sent, or if the preview=1 is not sent, the glasses will reject the setting.
Header 	Packet size 	Pad 	Seq 	SubCommand 	Preview 	Height 	Depth
26 	08 	00 	00-FF 	02 	00/ 01 	00-08 	01-09
Button Double Tap Settings (0x06)
Header 	Packet size 	Pad 	Seq 	SubCommand 	Action
26 	06 	00 	00-FF 	04 	XX
Actions
Action ID 	Description
00 	None (Close Active Feature)
01 	
02 	Open Translate
03 	Open Teleprompter
04 	Show Dashboard
05 	Open Transcribe
Button Long Press Settings (0x07)
Header 	Packet size 	Pad 	Seq 	SubCommand 	Enable/Disable
26 	06 	00 	00-FF 	07 	00 / 01
Actions
Action ID 	Description
00 	Disabled
01 	Enabled
Activate Microphone on Head Lift Settings (0x08)

Enables or Disables streaming audio to the phone when the head is lifted..
Header 	Packet size 	Pad 	Seq 	SubCommand 	Enable/Disable (inverted)
26 	06 	00 	00-FF 	08 	01 / 00
Hardware Get (0x3F)

TODO See: Button Double Tap Settings (0x06)
05 	Get Double Tap Action
06 	Get ble set lum coeffic
07 	Get Enable/Disable Long Press Action
08 	Get Activate Mic on Head Up
Hardware Display Get (0x3B)

Fetches the current screen height and depth values. Send to Right side.
Command
3B
Response
Header 	Success 	Height 	Depth
3B 	C9 	00 ~ 08 	01 ~ 09
Head Up Angle Set (0x0B)

Sets the angle at which the display turns on when the wearer looks up. Send to Right side. Response is generic
Header 	Angle 	Level?
0B 	00 ~ 3C 	01 ?
Head Up Action Set (0x08)

Sets what happens when the user lifts their head up. Send to Both sides. Response is generic If the Local / Global is set to 03, the command can be sent to just the left and left will forward to the right? If set to 04, the command sends to both?
Header 	Packet size 	Pad 	Seq 	Local / Global 	Action
08 	06 	00 	00-FF 	03 / 04 	00 / 02
Modes
Action ID 	Action
00 	Show the Dashboard
01 	??
02 	Do Nothing
Head Up Angle Get (0x32)

Send to Right side.
Command
32
Response
Header 	Success 	Angle
32 	C9 	00 ~ 42
Head Up Calibration Control (0x10)

Set or clear the base horizontal head up angle. Response is generic TODO
Subcommands

TODO
Subcommand ID 	Description
01 	Clear
02 	Set
Info Battery and Firmware Get(0x2C)

Get some basic info about the device.
Command 	Subcommand
2C 	01 / 02
Response
Header 	Model (ASCII) 	Left Battery 	Right Battery 	?? 	L Major Version 	L Minor Version 	L Sub Version 	R Major Version 	R Minor Version 	R Sub Version
2C 	A / B 	00~64 	00 ~ 64 	00 00 00 	01 	06 	03 	01 	06 	03
Info MAC Address Get (0x2D)

TODO
Info Serial Number Lens Get (0x33)

Fetches the Serial number of the individual lens.
Command
33
Response

Response is ASCII
Header 	Magic 	ASCII Payload
33 (“3”) 	33 (“3”) 	47 31 52 31 45 4b 54 30 39 39 03
Info Serial NumberGlasses Get (0x34)

Fetched the Serial number of the Glasses.
Command
34
Response

Response is ASCII
Header 	Magic 	Frame Type 	Frame Color 	ID
34 (“4”) 	34 (“4”) 	SXXX 	LXX 	LXXXXXX

The serial number can be fetched from the BLE Scan Result object. For example:S110LAAL103842 The first 4 characters indicate the frame shape.
Frame 	Code 	Description
S100 	A 	Round
S110 	B 	Square

The next 3 characters indicate the color of the frames.
Color 	Code 	Description
LAA 	Grey1 	Grey
LBB 	Brown1 	Brown
LCC 	Green1 	Green
Info ESB Channel Get (0x35)

The two lenses communicate using Enhanced ShockBurst (ESB). This lists the ESB channel they are talking on.
Command
35
Response
Header 	Success / Fail 	Channel ID
35 	C9 / CA 	57
Info ESB Channel Notification Count Get (0x36)

The two lenses communicate using Enhanced ShockBurst (ESB). Fetches the number of pending ESB notifications.
Command
36
Response
Header 	Success / Fail 	Notification Count
36 	C9 / CA 	02
Info Time Since Boot Get (0x37)

Fetches the time since boot in seconds.
Command
37
Response
Header 	Payload 	??
37 	49 1a 00 00 	00 / 01
Info Buried Point Get (0x3E)

Fetch buried point data, which is essentially user usage tracking: https://www.php.cn/faq/446290.html
Command
3E
Response
Header 	Success 	JSON Payload
3E 	C9 	Example censored as it may contain private data.
Language Set (0x3D)

Set the language for the glasses.
Command 	Size 	Pad 	Sequence 	Magic 	Language ID
3D 	06 	00 	XX 	01 	XX
Language ID
ID 	Language
01 	Chinese
02 	English
03 	Japanese
04 	??
05 	French ?
06 	German
07 	Spanish ?
0E 	Italian
Response
TODO
Silent Mode Set (0x03)

This will activate or deactivate the silent mode of the glasses. Send to Both sides. Response is generic.
Command 	Enable/Disable 	Description
03 	0C 	Silent Mode On
03 	0A 	Silent Mode Off
Silent Mode Get (0x2B)

Fetches the current state of the silent mode setting. Additionally contains the State of the glasses. Send to Both sides.
Command
2B
Response
Header 	?? 	Silent Enabled 	State Code
2B 	69 	0C (true) / 0A (false) 	XX
State Code

These codes are the same sent with the F5 device events.
Code 	State
06 	Glasses worn
07 	Glass not worn
08 	Glass in case, lid open
0A 	Glasses in case, lid closed
0B 	Glasses in case, lid closed, case plugged in
Microphone Set (0x0E)

Turn the microphone on or off. Send to Left side. Response is generic
Header 	Enable / Disable
0E 	00 / 01
MTU Set (0x4D)

Sets the Device BLE MTU value. This should match the setting requested from the host OS ble stack, but never exceed 251. Send to Both sides. Response is generic.
Command 	MTU value
4D 	FB (251)
Navigation Control (0x0A)

TODO
Subcommand
Subcommand ID 	Description
00 	Init
01 	Update Trip Status
02 	Update Map Overview
03 	Set Panoramic Map
04 	App Sync Packet
05 	Exit
06 	Arrived
Notification App List Set (0x04)

Send over the config for notifications. This has JSON with some booleans and an allowlist for which apps to display notifications for. The easiest way to use this on android is to set a single app as allowed and just put all notifications under that particular app id. Send to Left side. Response is generic.
Command 	Chunk Count 	Chunk 	Payload (JSON) (Max size 180 bytes)
04 	01 ~ FF 	00 ~ FF 	{"calendar_enable":true, "Call_enable":true, "Msg_enable":true, "Ios_mail_enable":true, "app":{   "List":[     {"id":"com.app",      "name":"App Name"}, ...],   “enable”:true }
Notification App List Get (0x2E)

TODO
Notification Apple Specific Get (0x38)

Get the ANCS settings.

TODO
Notification Auto Display Set (0x4F)

Enables or disables the notification auto displaying on the glasses and the amount of time that the notification will be shown. Send to Both sides. Response is generic.
Command 	Enable/Disable 	Display Timeout (seconds)
4F 	00 ~ 01 	00 ~ FF
Notification Auto Display Get (0x3C)

Fetches the current value, in seconds, for how long a notification will be shown to the user when it arrives.. Send to Left side.
Command
3C
Response
Header 	Success 	Enabled 	Timeout
3C 	C9 	00 / 01 	00 ~ FF
Notification Send (0x4B)

Send a notification to the glasses. Divide the JSON payload into 180 byte segments and determine the number of chunks. Send multiple packets one for each chunk. The chunk count should be the same in all packets. The Chunk Index of each packet should increment by 1 as they are sent. Wait for a C9 (success) response before sending the next chunk. The msg_id in the JSON should be unique for each notification sent. Send to Left side. Response is generic.
Command 	Pad 	Chunk Count 	Chunk Index 	Payload (JSON) (Max size 180 bytes)
4B 	00 	01 ~ FF 	00 ~ FF 	{"ncs_notification":{"msg_id":16,"action":0,"app_identifier":"nodomain.freeyourgadget.gadgetb","title":"Test notification","subtitle":"","message":"This is a test notification from Gadgetbridge","time_s":1749606217,"date":"2025-06-10 18:43:37","display_name":"Gadgetbridge"}}
Notification Clear (0x4C)

Clear a notification on the glasses. The msg_id should match one that was sent via the 0x4B command. Send to Left side. Response is generic.
Command 	Payload (32-bit msg id)
4C 	00 00 00 1b
Status Get (0x22)

TODO
Status Running App Get (0x39)

TODO
System Control (0x23)
Debug Logging Set (0x23 0x6C)

Enables debug logging on the glasses. Response is generic. Oddly enough, 00 is enabled while C1 is disabled. See: Debug (0xF4).
Command 	Subcommand 	Enable/Disable
23 	6C 	00 / C1
Reboot (0x23 0x72)

Reboots the glasses. No response.
Command 	Subcommand
23 	72
Firmware Build Info Get (0x23 0x74)

Send and the device should respond with fw build information
Command 	Subcommand
23 	74
Response

This response kind of sucks because it has no header and is just raw ASCII data. Currently it always starts with “net”.
ASCII Payload
net build time: 2024-12-28 20:21:57, app build time 2024-12-28 20:20:45, ver 1.4.5, JBD DeviceID 4010
Teleprompter Control (0x09)

TODO
Subcommand
Subcommand ID 	Description
01 	Init / Set Text
02 	Set Position
03 	Update Text
04 	None
05 	Exit
Teleprompter Suspend (0x24)

TODO
Teleprompter Position Set(0x25)

TODO
Text Set (0x4E)

Display full screen text. The official app uses this to show the response for the AI voice prompt. This can also be used to show just the text without the AI context. Send to Both sides.
Command 	Size 	Sequence 	Chunk Count 	Pad 	Chunk 	Pad 	Display Style 	Canvas State 	Position 	Page 	Page Count 	Payload (UTF-8)
4E 	XX 	00 ~ FF 	01 ~ FF 	00 	01 ~ FF 	00 	X 	X 	XX XX 	XX 	XX 	XX XX ...
Display Style

The display style is only 4bits, not a full byte.
Display Style ID 	Description
0 	??
1 	??
2 	??
3 	Even AI displaying / Auto Scroll
4 	Even AI Complete
5 	Even AI displaying / Manual Scroll
6 	Even AI Network Error
7 	Show Text only
Canvas State

The Canvas State is only 4bits, not a full byte.
Canvas State 	Description
0 	Draw to existing Canvas
1 	Start New Canvas
Timer Control (0x07)

Is this a stop watch or timer?!?! TODO
Transcribe Control (0x0D)

TODO
Translate Control (0x0F)

TODO
Tutorial Control (0x1F)

TODO This has subcommands that need to be sent in sequence.
Unpair (0x47)

Removes the current Host (phone or PC) mac address from the glasses as a paired device. You should probably just unpair from the host device instead. Do NOT send this command unless you REALLY know what you are doing. Send to Both sides. Response is generic.
Command
47
Upgrade Control (0x17)

Before a fw update, and after a successful Firmware Update, resources will be allocated to the DFU firmware image. This command erases the image. Do NOT send this command unless you REALLY know what you are doing. Send to Both sides. Response is generic.
Command
17
Wear Detection Set (0x27)

Enable or disable Wear Detection. When enabled, additional 0xF5 messages are sent when worn or not. Response is generic.
Command 	Enable/Disable
27 	00 ~ 01
Wear Detection Get (0x3A)

Fetches the current configured option for wear detection. This enables or disables a proximity sensor on the glasses that can determine if the glasses are worn or not. Send to Either side.
Command
3A
Response
Header 	Success 	Silent Enabled
3A 	C9 	00 / 01
?? (0x50)

TODO. Send to Both sides.
Command 	Size 	?? 	?? 	?? 	??
50 	06 	00 	00 	01 	01
Response

The full command is repeated back.
Command 	Size 	?? 	?? 	?? 	??
50 	06 	00 	00 	01 	01
Scratch
Pane Map Set (0x07)
Command 	Length 	Pad 	Sequence
06 	XX 	00 	00 ~ FF
Subcommand 	Chunk Count 	Pad 	Chunk 	Pad 			
07 	01 ~ FF 	00 	01 ~ FF 	00 			

[hand out map] | 7:54:14 372ms | requestStaticImage cost 862ms bytes:180604 [BLE] | 7:54:14 456ms | BleManager receive cmd: R22, len: 10, rssi: -66, data = 22 0a 00 00 01 00 01 00 04 00 [dashboard] | 7:54:14 456ms | _observeGlassInfo data = 22 0a 00 00 01 00 01 00 04 00 [dashboard] | 7:54:14 456ms | _observeGlassInfo unread:0 lowPower:true combine:DashboardCombineMode.full widget:DashboardMainWidgetMode.handOutMap index:0 [hand out map] | 7:54:14 652ms | handleImageDataAsync cost 280ms [debug] | 7:54:15 142ms | offsetWithPosition vertical:183 horizontal:400 size:800.0 366.0 x:0.5 y:0.5 [BLE] | 7:54:15 145ms | BleManager send cmd: L06, len: 12, data = 06 0c 00 57 07 01 00 01 00 00 04 00

[proto] | 7:54:15 265ms | configCityWalkLoading result:true cost 130ms [debug] | 7:54:15 531ms | offsetWithPosition vertical:183 horizontal:400 size:800.0 366.0 x:0.5 y:0.5 [BLE] | 7:54:15 526ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 58 07 1c 00 01 00 00 04 02 01 00 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00 00 00 70 00 00 00 00 00 04 00 00 00 00 00 00 00 0f 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00 00 00 60 00 00 00 00 00 04 00 00 00 00 00 00 80 07 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00 00 00 c0 00 00 00 00 00 04 00 00 00 00 00 00 c0 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 01 00 00 00 00 7c 00 00 00 00 00 00 e0 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 03 00 00 00 00 f0 00 00 00 [BLE] | 7:54:15 598ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 59 07 1c 00 02 00 00 00 00 70 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 00 00 00 00 80 01 00 00 00 00 00 78 00 00 00 00 00 00 00 00 00 00 00 00 00 80 00 00 00 00 00 00 00 00 00 00 06 00 00 00 00 00 03 00 00 00 00 00 3c 00 00 00 00 80 00 00 00 00 00 00 00 00 c0 00 00 00 00 00 00 00 00 00 00 0c 00 00 00 00 00 02 00 00 00 00 00 1c 00 00 00 00 f0 00 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00 00 00 00 18 00 00 00 00 00 03 00 00 00 00 00 0e 00 00 00 00 3f 00 00 00 00 00 00 00 00 10 00 00 00 00 00 00 00 00 00 00 30 00 00 00 [BLE] | 7:54:15 620ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 5a 07 1c 00 03 00 00 00 03 00 00 00 00 00 07 00 00 00 e0 07 00 00 00 00 00 00 00 00 08 00 00 00 00 00 00 00 00 00 00 30 00 00 00 00 00 03 00 00 00 00 80 03 00 00 00 fc 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 60 00 00 00 00 00 03 00 00 00 00 c0 03 00 00 00 0f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00 00 00 01 00 00 00 00 e0 01 00 00 c0 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 01 00 00 00 00 01 00 00 00 00 e0 00 00 00 e0 00 00 00 00 00 00 00 1c 00 00 00 00 00 00 00 00 00 00 00 00 [BLE] | 7:54:15 661ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 5b 07 1c 00 04 00 00 80 03 00 00 00 80 01 00 00 00 00 70 00 00 00 38 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 00 00 00 80 01 00 00 00 00 38 00 00 00 1c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 06 00 00 00 80 00 00 00 00 00 38 00 00 00 07 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0c 00 00 00 80 00 00 00 00 00 1c 00 00 80 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 18 00 00 00 c0 00 00 00 00 00 1c 00 00 c0 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 [BLE] | 7:54:15 661ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 5c 07 1c 00 05 00 00 00 00 00 00 00 00 38 00 00 00 c0 00 00 00 00 00 0e 00 00 60 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 30 00 00 00 c0 00 00 00 00 00 07 00 00 30 00 00 00 00 00 00 00 18 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 60 00 00 00 60 00 00 00 00 00 07 00 00 18 00 00 00 00 f0 00 00 3e 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00 60 00 00 00 00 80 03 00 00 18 00 00 00 00 fe 1f 80 37 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 01 00 00 30 00 00 00 00 80 03 00 00 0c 00 00 00 c0 07 fc fd 21 00 00 00 [BLE] | 7:54:15 688ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 5d 07 1c 00 06 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 00 00 30 00 00 00 00 c0 01 00 00 06 00 00 00 f0 18 c0 bf 61 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 07 00 00 18 00 00 00 00 e0 00 00 00 07 00 00 00 18 30 00 80 31 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 06 00 00 0c 00 00 00 00 e0 00 00 00 03 00 00 00 0e c0 01 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0c 00 00 06 00 00 00 00 70 00 00 80 01 00 00 00 07 00 07 00 0f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 18 00 80 03 00 00 00 00 70 00 00 c0 00 00 00 00 01 00 38 [BLE] | 7:54:15 741ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 5e 07 1c 00 07 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 70 00 e0 01 00 00 00 00 38 00 00 c0 00 00 00 00 00 00 c0 0f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 e0 01 78 00 00 00 00 00 18 00 00 60 00 00 00 00 00 00 00 18 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0c 00 00 80 ff 1f 00 00 00 00 00 1c 00 00 30 00 00 00 00 00 00 00 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 00 00 00 fc 03 00 00 00 00 00 0e 00 00 38 00 00 00 00 00 00 00 20 00 00 00 00 00 00 40 00 00 00 10 00 00 00 30 00 00 00 60 0f 00 00 00 00 00 0e 00 00 18 00 00 [BLE] | 7:54:15 751ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 5f 07 1c 00 08 00 00 00 00 00 00 20 00 00 00 00 00 00 80 00 00 00 20 00 00 00 60 00 00 00 60 03 00 00 00 00 00 07 00 00 0c 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00 01 00 00 20 00 00 00 40 00 00 00 60 03 00 00 00 00 80 03 00 00 06 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00 02 00 00 40 00 00 00 80 00 00 00 60 03 00 00 00 00 80 03 00 00 06 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00 08 00 00 80 00 00 00 80 01 00 00 60 03 00 00 00 00 c0 01 00 00 03 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00 30 00 00 00 00 00 00 40 02 00 00 60 03 00 00 00 00 c0 01 [BLE] | 7:54:15 956ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 60 07 1c 00 09 00 00 80 01 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00 80 01 00 50 00 00 00 20 06 00 00 60 02 00 00 00 00 e0 00 00 80 01 00 00 00 00 00 00 00 c0 00 00 00 00 00 00 00 00 1c 00 08 00 00 00 10 0c 00 00 60 02 00 00 00 00 70 00 00 c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 ff 0f 00 00 00 08 08 00 00 60 02 00 00 00 00 70 00 00 c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 06 10 00 00 60 02 00 00 00 00 38 00 00 60 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 20 00 00 60 02 00 [BLE] | 7:54:15 966ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 61 07 1c 00 0a 00 00 00 00 38 00 00 60 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 40 00 00 60 02 00 00 00 00 1c 00 00 30 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 00 c0 01 00 60 02 00 00 00 00 0e 00 00 30 00 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 60 00 80 03 00 60 02 00 00 00 00 0e 00 00 18 00 00 00 00 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 30 00 00 0e 00 60 02 00 00 00 00 07 00 00 08 00 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 18 00 00 [BLE] | 7:54:15 966ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 62 07 1c 00 0b 00 1c 00 60 02 00 00 00 00 07 00 00 0c 00 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0c 00 00 70 00 60 02 00 00 00 80 03 00 00 04 00 00 00 80 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 02 00 00 c0 03 60 02 00 00 00 80 01 00 00 06 00 00 00 80 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 1f 60 02 00 00 00 c0 01 00 00 03 00 00 00 80 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 00 00 00 fc 63 02 00 00 00 c0 01 00 00 03 00 00 00 80 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 [BLE] | 7:54:16 173ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 63 07 1c 00 0c 00 00 00 06 00 00 00 80 ff ff 07 00 00 e0 00 00 80 01 00 00 00 80 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0c 00 00 00 00 f0 ff ff 00 00 70 00 00 80 01 00 00 00 c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 18 00 00 00 00 00 00 e0 0f 00 70 00 00 80 01 00 00 00 c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 30 00 00 00 00 00 00 00 3c 00 38 00 00 80 01 00 00 00 c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00 f0 00 38 00 00 80 01 00 00 00 40 00 01 00 00 00 00 00 00 00 00 00 [BLE] | 7:54:16 173ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 64 07 1c 00 0d 00 00 00 00 00 00 00 00 c0 00 00 00 00 00 00 00 c0 03 1c 00 00 80 01 00 00 00 60 00 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 00 00 00 00 00 00 00 00 1f 1c 00 00 80 00 00 00 00 60 00 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00 7c 0e 00 00 80 00 00 00 00 60 00 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00 e0 0f 00 00 80 00 00 00 00 60 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00 80 07 00 00 80 00 00 00 00 30 00 00 00 00 00 00 [BLE] | 7:54:16 354ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 65 07 1c 00 0e 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 04 00 00 00 00 00 00 00 3f 00 00 80 00 00 00 00 30 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 3e 00 00 00 00 00 00 80 f3 00 00 ff ff ff 0f 00 30 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 30 41 00 00 00 00 00 00 80 c3 ff ff ff ff ff ff 0f 30 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 80 00 00 00 00 00 00 80 01 fe 1f 00 00 00 80 ff 1f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40 00 3f 00 00 00 00 00 c0 01 00 00 00 00 00 00 80 ff 03 [BLE] | 7:54:16 414ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 66 07 1c 00 0f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 00 00 ff ff 3f 00 00 c0 01 00 00 00 00 00 00 00 80 ff 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 f0 03 00 e0 00 00 00 00 00 00 00 00 00 f0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 00 02 00 e0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 02 00 00 00 00 02 00 60 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0e 00 00 00 00 02 00 70 00 00 00 00 00 [BLE] | 7:54:16 445ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 67 07 1c 00 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 38 00 00 00 00 02 00 70 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00 00 02 00 30 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 00 00 00 02 00 30 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0c 00 00 00 02 00 38 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 30 00 00 00 02 00 38 [BLE] | 7:54:16 472ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 68 07 1c 00 11 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00 02 00 38 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 00 00 02 00 18 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0c 00 00 02 00 18 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 00 00 02 00 1c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 60 [BLE] | 7:54:16 489ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 69 07 1c 00 12 00 00 00 02 00 1c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 00 00 02 00 1c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 01 00 02 00 0c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 00 02 00 0c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 04 00 02 00 0e 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 [BLE] | 7:54:16 517ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 6a 07 1c 00 13 00 00 00 00 00 00 08 00 02 00 1f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 00 02 00 1f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 00 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 60 00 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40 00 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 [BLE] | 7:54:16 545ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 6b 07 1c 00 14 00 00 00 00 00 00 00 00 00 00 00 80 00 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 01 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 ff 07 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 02 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 01 fc 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 06 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 7e 00 [BLE] | 7:54:16 568ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 6c 07 1c 00 15 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 04 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 80 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 1c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 e0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 80 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 30 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 [BLE] | 7:54:16 605ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 6d 07 1c 00 16 00 00 3f 00 00 00 0c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 02 80 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 e1 1f 00 00 18 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40 02 00 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 01 e0 01 00 60 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40 02 00 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 3e 00 80 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 02 00 1b 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 e0 03 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 03 00 1b 00 00 00 00 00 00 00 [BLE] | 7:54:16 624ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 6e 07 1c 00 17 00 00 00 00 00 00 00 01 00 00 1e 00 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 00 1b 00 00 00 00 00 00 00 00 00 00 00 00 fe 7f 00 00 30 00 04 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0e 00 1b 00 00 00 00 00 00 00 00 00 00 00 fc ff ff 7f 00 c0 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f8 ff 1b 00 00 00 00 00 00 00 00 00 00 c0 1f 80 00 f8 0f 00 03 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 00 00 00 00 00 00 00 f8 01 80 00 00 7e 00 06 20 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 [BLE] | 7:54:16 662ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 6f 07 1c 00 18 00 00 00 00 00 00 00 00 00 3e 00 80 01 00 e0 03 08 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 00 00 00 00 00 00 00 0f 00 40 02 00 00 0f 30 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 00 00 00 00 00 00 00 11 00 20 fe 00 00 3c 40 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 00 00 00 00 00 00 00 20 00 10 f0 ff 1f 70 c0 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 00 00 00 00 00 00 00 40 00 08 18 60 e0 c0 80 21 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 [BLE] | 7:54:16 693ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 70 07 1c 00 19 00 00 00 1b 00 00 00 00 00 00 00 00 00 00 80 00 04 0c 30 c0 81 01 31 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 00 00 00 00 00 00 00 00 01 02 06 18 20 03 07 1a 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 00 00 00 00 00 00 00 00 03 03 0b 0c 10 06 06 0e 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 00 00 00 00 00 00 00 00 06 86 11 06 18 0c 0c 04 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 fb ff 07 00 00 00 00 00 00 00 00 00 0c cc 32 03 04 18 0c 03 00 00 00 00 00 00 00 00 00 00 [BLE] | 7:54:16 737ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 71 07 1c 00 1a 00 00 00 00 00 00 00 00 fb ff ff 00 00 00 00 00 00 00 00 00 18 78 cc 01 02 18 d8 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 c0 07 00 00 00 00 00 00 00 00 30 b0 cc 00 01 64 78 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 1e 00 00 00 00 00 00 00 00 60 0c 73 80 00 42 38 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f0 00 00 1b 00 00 38 00 00 00 00 00 00 00 00 40 0c 23 40 00 81 3c 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 7f 00 1b 00 00 70 00 00 00 00 00 00 00 00 80 98 14 20 80 00 a3 00 00 00 00 00 00 [BLE] | 7:54:16 761ms | BleManager send cmd: L06, len: 189, data = 06 bd 00 72 07 1c 00 1b 00 00 00 00 00 00 00 00 00 00 00 00 7f 1b 00 00 c0 00 00 00 00 00 00 00 00 00 a1 08 10 40 80 61 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 80 1b 00 00 80 01 00 00 00 00 00 00 00 00 d2 04 08 20 c0 62 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 03 00 00 00 00 00 00 00 00 a4 0d 06 10 60 d2 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 03 00 00 00 00 00 00 00 00 48 15 02 08 30 ce 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 06 00 00 00 00 00 00 00 00 50 a0 01 04 08 84 01 [BLE] | 7:54:16 786ms | BleManager send cmd: L06, len: 185, data = 06 b9 00 73 07 1c 00 1c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 0c 00 00 00 00 00 00 00 00 30 c1 00 02 0c 8c 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 0c 00 00 00 00 00 00 00 00 60 80 00 01 06 89 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1b 00 00 00 18 00 00 00 00 00 00 00 00 80 00 c1 00 41 0c 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 3b 00 00 00 18 00 00 00 00 00 00 00 00 00 01 66 80 60 12 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 3b 00 00 00 10 00 00 00 00 00 00 [proto] | 7:54:16 919ms | configCityWalkMapData result:true cost 1399ms
Send Arrow Data

[hand out map] | 7:54:16 919ms | _sendMapData result:true [debug] | 7:54:16 919ms | offsetWithPosition vertical:183 horizontal:400 size:800.0 366.0 x:0.5 y:0.5 [hand out map] | 7:54:16 919ms | _calculateArrowPosition x:132 y:52 [hand out map] | 7:54:16 921ms | prepareArrowData image 32 32 3072 128 [hand out map] | 7:54:16 921ms | _sendArrowData bytes:128 heading:55 x:132 y:52 [BLE] | 7:54:16 931ms | BleManager send cmd: L06, len: 144, data = 06 90 00 74 07 01 00 01 00 00 04 03 84 00 34 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 00 00 00 f0 03 00 00 fc 3f 00 00 fe 7f 00 00 fc 3f 00 00 00 3f 00 00 00 3c 00 00 00 3e 00 00 00 3e 00 00 00 1e 00 00 00 3e 00 00 00 1e 00 00 00 0e 00 00 00 06 00 00 00 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 [proto] | 7:54:17 23ms | configCityWalkArrowData result:true cost 101ms

[debug] | 7:54:48 40ms | offsetWithPosition vertical:183 horizontal:400 size:800.0 366.0 x:0.5 y:0.5 [hand out map] | 7:54:48 40ms | _calculateArrowPosition x:132 y:52 [hand out map] | 7:54:48 41ms | prepareArrowData image 32 32 3072 128 [hand out map] | 7:54:48 41ms | _sendArrowData bytes:128 heading:90 x:132 y:52 [BLE] | 7:54:48 46ms | BleManager send cmd: L06, len: 144, data = 06 90 00 7d 07 01 00 01 00 00 04 03 84 00 34 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 f8 01 00 00 f0 03 00 00 e0 07 00 00 c0 0f 00 00 80 1f 00 00 00 3f 00 00 00 7e 00 00 00 7e 00 00 00 3f 00 00 80 1f 00 00 c0 0f 00 00 e0 07 00 00 f0 03 00 00 f8 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 [debug] | 7:54:48 48ms | offsetWithPosition vertical:183 horizontal:400 size:800.0 366.0 x:0.5 y:0.5 [proto] | 7:54:48 162ms | configCityWalkArrowData result:true cost 121ms

07 01 00 01 00 00 04 03 84 00 34 00

0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000011111000 0000000100000000 0000000011110000 0000001100000000 0000000011100000 0000011100000000 0000000011000000 0000111100000000 0000000010000000 0001111100000000 0000000000000000 0011111100000000 0000000000000000 0111111000000000 0000000000000000 0111111000000000 0000000000000000 0011111100000000 0000000010000000 0001111100000000 0000000011000000 0000111100000000 0000000011100000 0000011100000000 0000000011110000 0000001100000000 0000000011111000 0000000100000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000
Pad 	?? 	Layer? 	X Offset 	Y Offset 	
00 	04 	00 ~ 04 	84 00 	34 00 	

[BLE] | 7:55:05 118ms | BleManager send cmd: R06, len: 144, data = 06 90 00 86 07 01 00 01 00 00 04 03 84 00 34 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 [BLE] | 7:55:05 158ms | BleManager receive cmd: R06, len: 10, rssi: -58, data = 06 90 00 86 07 01 00 01 00 00 [proto] | 7:55:05 158ms | configCityWalkArrowData result:true cost 103ms

06 90 00 86 07 01 00 01 00 00 04 03 84 00 34 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00 e0 00 00 00 f0 00 00 00 78 00 00 00 f0 00 00 00 f0 00 00 00 78 00 00 00 f0 00 00 00 f0 00 01 00 f8 ff 01 00 f8 ff 00 00 f0 7f 00 00 f0 3f 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00

Set the dashboard mode, currently has three modes: minimal, dual, full
Subcommand 	Chunk Count 	Pad 	Chunk 	Payload
07 	01 ~ FF 		00 00 	07  01 00  01 00  00 04 03   07  1c 00  01 00  00 04 02 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000011111000 0000000100000000 0000000011110000 0000001100000000 0000000011100000 0000011100000000 0000000011000000 0000111100000000 0000000010000000 0001111100000000 0000000000000000 0011111100000000 0000000000000000 0111111000000000 0000000000000000 0111111000000000 0000000000000000 0011111100000000 0000000010000000 0001111100000000 0000000011000000 0000111100000000 0000000011100000 0000011100000000 0000000011110000 0000001100000000 0000000011111000 0000000100000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000 0000000000000000
Head Up Calibration Control (0x10)

Set or clear the base horizontal head up angle. Response is generic TODO ??

Reset 0degree position 2025-05-18 18:45:07.339 10946-10996 ble_L com.even.g1 E sent status 10 L 0 , sent size: 5 sent: 10 05 00 04 01 2025-05-18 18:45:07.343 10946-10996 ble_R com.even.g1 E sent status 10 R 0 , sent size: 5 sent: 10 05 00 04 01 2025-05-18 18:45:07.375 10946-12531 ble_L com.even.g1 E received L len: 20,--10 06 00 04 01 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00

Calibration 2025-05-18 18:47:39.362 10946-10993 ble_L com.even.g1 E sent status 39 L 0 , sent size: 5 sent: 39 05 00 5f 01 2025-05-18 18:47:39.447 10946-12531 ble_L com.even.g1 E received L len: 6,--39 05 00 5f 01 00 2025-05-18 18:47:39.447 10946-12531 ble_L com.even.g1 W callback type: Receive, data: [B@70545f6 2025-05-18 18:47:39.448 12038-12732 nodomain.f....BtLEQueue nod...n.freeyourgadget.gadgetbridge D characteristic changed: 6e400003-b5a3-f393-e0a9-e50e24dcca9e value: 0x39 0x05 0x00 0x5f 0x01 0x00 2025-05-18 18:47:39.449 12038-12732 nodomain.f...ideManager nod...n.freeyourgadget.gadgetbridge D Unhandled payload on side 0: 0x39 0x05 0x00 0x5f 0x01 0x00 2025-05-18 18:47:39.456 10946-10993 ble_R com.even.g1 E sent status 39 R 0 , sent size: 5 sent: 39 05 00 5f 01 2025-05-18 18:47:39.501 10946-12530 ble_R com.even.g1 E received R len: 6,--39 05 00 5f 01 00 2025-05-18 18:47:39.501 10946-12530 ble_R com.even.g1 W callback type: Receive, data: [B@ad3edf7 2025-05-18 18:47:39.503 12038-12732 nodomain.f....BtLEQueue nod...n.freeyourgadget.gadgetbridge D characteristic changed: 6e400003-b5a3-f393-e0a9-e50e24dcca9e value: 0x39 0x05 0x00 0x5f 0x01 0x00 2025-05-18 18:47:39.506 12038-12732 nodomain.f...ideManager nod...n.freeyourgadget.gadgetbridge D Unhandled payload on side 1: 0x39 0x05 0x00 0x5f 0x01 0x00 2025-05-18 18:47:39.512 10946-10993 ble_R com.even.g1 E sent status 50 R 0 , sent size: 6 sent: 50 06 00 00 01 01 2025-05-18 18:47:39.801 10946-12530 ble_R com.even.g1 E received R len: 6,--50 06 00 00 01 01 2025-05-18 18:47:39.801 10946-12530 ble_R com.even.g1 W callback type: Receive, data: [B@2c7c464 2025-05-18 18:47:39.803 12038-12732 nodomain.f....BtLEQueue nod...n.freeyourgadget.gadgetbridge D characteristic changed: 6e400003-b5a3-f393-e0a9-e50e24dcca9e value: 0x50 0x06 0x00 0x00 0x01 0x01 2025-05-18 18:47:39.807 12038-12732 nodomain.f...ideManager nod...n.freeyourgadget.gadgetbridge D Unhandled payload on side 1: 0x50 0x06 0x00 0x00 0x01 0x01 2025-05-18 18:47:39.814 10946-10993 ble_L com.even.g1 E sent status 10 L 0 , sent size: 7 sent: 10 07 00 0c 02 01 00 2025-05-18 18:47:39.822 10946-10993 ble_R com.even.g1 E sent status 10 R 0 , sent size: 7 sent: 10 07 00 0c 02 01 00 2025-05-18 18:47:39.850 10946-12531 ble_L com.even.g1 E received L len: 20,--10 06 00 0c 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:47:39.850 10946-12531 ble_L com.even.g1 W callback type: Receive, data: [B@63d46cd 2025-05-18 18:47:39.856 12038-12732 nodomain.f....BtLEQueue nod...n.freeyourgadget.gadgetbridge D characteristic changed: 6e400003-b5a3-f393-e0a9-e50e24dcca9e value: 0x10 0x06 0x00 0x0c 0x02 0xc9 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 2025-05-18 18:47:39.862 12038-12732 nodomain.f...ideManager nod...n.freeyourgadget.gadgetbridge D Unhandled payload on side 0: 0x10 0x06 0x00 0x0c 0x02 0xc9 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 2025-05-18 18:47:39.889 10946-12530 ble_R com.even.g1 E received R len: 20,--10 06 00 0c 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00

Ack on glasses 2025-05-18 18:48:41.449 10946-12530 ble_R com.even.g1 E received R len: 20,--10 06 00 0d 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00

Done in app E sent status 10 L 0 , sent size: 7 sent: 10 07 00 05 02 00 00 2025-05-18 18:53:19.222 13285-13467 ble_R com.even.g1 E sent status 10 R 0 , sent size: 7 sent: 10 07 00 05 02 00 00 2025-05-18 18:53:19.269 13285-13632 ble_R com.even.g1 E received R len: 20,--10 06 00 05 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.269 13285-13632 ble_R com.even.g1 W callback type: Receive, data: [B@f8f1fff 2025-05-18 18:53:19.270 13285-13552 ble_R com.even.g1 E received R len: 20,--10 06 00 05 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.270 13285-13732 ble_R com.even.g1 E received R len: 20,--10 06 00 05 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.270 13285-13732 ble_R com.even.g1 W callback type: Receive, data: [B@c2d9c15 2025-05-18 18:53:19.270 13285-13552 ble_R com.even.g1 W callback type: Receive, data: [B@9ea81cc 2025-05-18 18:53:19.276 13285-13586 ble_L com.even.g1 E received L len: 20,--10 06 00 05 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.276 13285-13733 ble_L com.even.g1 E received L len: 20,--10 06 00 05 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.276 13285-13586 ble_L com.even.g1 W callback type: Receive, data: [B@2380b2a 2025-05-18 18:53:19.276 13285-13733 ble_L com.even.g1 W callback type: Receive, data: [B@1bfbd1b 2025-05-18 18:53:19.276 13285-13631 ble_L com.even.g1 E received L len: 20,--10 06 00 05 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.276 13285-13631 ble_L com.even.g1 W callback type: Receive, data: [B@823eeb8 2025-05-18 18:53:19.338 13285-13737 ble_L com.even.g1 E Failed to write characteristic 201 2025-05-18 18:53:19.343 13285-13293 ble_L com.even.g1 E sent status 10 L 0 , sent size: 7 sent: 10 07 00 03 02 01 01 2025-05-18 18:53:19.349 13285-13293 ble_R com.even.g1 E sent status 10 R 0 , sent size: 7 sent: 10 07 00 03 02 01 01 2025-05-18 18:53:19.352 13285-13293 ble_L com.even.g1 E sent status 10 L 0 , sent size: 7 sent: 10 07 00 03 02 01 01 2025-05-18 18:53:19.525 13285-13738 ble_L com.even.g1 E Failed to write characteristic 201 2025-05-18 18:53:19.530 13285-13450 ble_L com.even.g1 E sent status 10 L 0 , sent size: 7 sent: 10 07 00 03 02 01 01 2025-05-18 18:53:19.535 13285-13450 ble_R com.even.g1 E sent status 10 R 0 , sent size: 7 sent: 10 07 00 03 02 01 01 2025-05-18 18:53:19.538 13285-13450 ble_L com.even.g1 E sent status 10 L 0 , sent size: 7 sent: 10 07 00 03 02 01 01 2025-05-18 18:53:19.806 13285-13586 ble_L com.even.g1 E received L len: 20,--10 06 00 02 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.806 13285-13733 ble_L com.even.g1 E received L len: 20,--10 06 00 02 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.806 13285-13586 ble_L com.even.g1 W callback type: Receive, data: [B@b930ff7 2025-05-18 18:53:19.806 13285-13631 ble_L com.even.g1 E received L len: 20,--10 06 00 02 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.806 13285-13733 ble_L com.even.g1 W callback type: Receive, data: [B@c853e64 2025-05-18 18:53:19.806 13285-13631 ble_L com.even.g1 W callback type: Receive, data: [B@31638cd 2025-05-18 18:53:19.817 13285-13586 ble_L com.even.g1 E received L len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.817 13285-13631 ble_L com.even.g1 E received L len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.817 13285-13586 ble_L com.even.g1 W callback type: Receive, data: [B@152d582 2025-05-18 18:53:19.817 13285-13631 ble_L com.even.g1 W callback type: Receive, data: [B@1deb493 2025-05-18 18:53:19.818 13285-13733 ble_L com.even.g1 E received L len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.818 13285-13733 ble_L com.even.g1 W callback type: Receive, data: [B@389dcd0 2025-05-18 18:53:19.833 13285-13586 ble_L com.even.g1 E received L len: 20,--10 06 00 06 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.833 13285-13586 ble_L com.even.g1 W callback type: Receive, data: [B@bda94c9 2025-05-18 18:53:19.833 13285-13631 ble_L com.even.g1 E received L len: 20,--10 06 00 06 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.833 13285-13631 ble_L com.even.g1 W callback type: Receive, data: [B@1dcc7ce 2025-05-18 18:53:19.833 13285-13733 ble_L com.even.g1 E received L len: 20,--10 06 00 06 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.833 13285-13733 ble_L com.even.g1 W callback type: Receive, data: [B@73606ef 2025-05-18 18:53:19.840 13285-13632 ble_R com.even.g1 E received R len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.841 13285-13732 ble_R com.even.g1 E received R len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.841 13285-13632 ble_R com.even.g1 W callback type: Receive, data: [B@3f7f5fc 2025-05-18 18:53:19.841 13285-13552 ble_R com.even.g1 E received R len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.841 13285-13552 ble_R com.even.g1 W callback type: Receive, data: [B@192da 2025-05-18 18:53:19.841 13285-13732 ble_R com.even.g1 W callback type: Receive, data: [B@a6eb485 2025-05-18 18:53:19.847 13285-13586 ble_L com.even.g1 E received L len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.847 13285-13586 ble_L com.even.g1 W callback type: Receive, data: [B@a06230b 2025-05-18 18:53:19.847 13285-13733 ble_L com.even.g1 E received L len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.847 13285-13733 ble_L com.even.g1 W callback type: Receive, data: [B@8cc75e8 2025-05-18 18:53:19.847 13285-13631 ble_L com.even.g1 E received L len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.847 13285-13631 ble_L com.even.g1 W callback type: Receive, data: [B@46b1401 2025-05-18 18:53:19.855 13285-13552 ble_R com.even.g1 E received R len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.855 13285-13732 ble_R com.even.g1 E received R len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00 2025-05-18 18:53:19.855 13285-13632 ble_R com.even.g1 E received R len: 20,--10 06 00 03 02 c9 00 00 00 00 00 00 00 00 00 00 00 00 00 00
Status Running App Get (0x39)

2025-06-10 18:49:51.248 4328-5106 ble_L com.even.g1 E sent status 39 L 0 , sent size: 5 sent: 39 05 00 97 01 2025-06-10 18:49:51.254 29819-30489 nodomain.f....BtLEQueue nod...n.freeyourgadget.gadgetbridge D characteristic changed: 6e400003-b5a3-f393-e0a9-e50e24dcca9e value: 0x4f 0xc9 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 2025-06-10 18:49:51.291 29819-30489 nodomain.f....BtLEQueue nod...n.freeyourgadget.gadgetbridge D characteristic changed: 6e400003-b5a3-f393-e0a9-e50e24dcca9e value: 0x39 0x05 0x00 0x97 0x01 0x00 2025-06-10 18:49:51.292 4328-30265 ble_L com.even.g1 E received L len: 6,--39 05 00 97 01 00 2025-06-10 18:49:51.294 29819-30489 nodomain.f...ideManager nod...n.freeyourgadget.gadgetbridge D Unhandled payload on side 0: 0x39 0x05 0x00 0x97 0x01 0x00 2025-06-10 18:49:51.295 4328-4328 BleCmd com.even.g1 I methodHandler: send, cmd: 39 len: 5 2025-06-10 18:49:51.300 4328-5106 ble_R com.even.g1 E sent status 39 R 0 , sent size: 5 sent: 39 05 00 97 01 2025-06-10 18:49:51.344 4328-30264 ble_R com.even.g1 E received R len: 6,--39 05 00 97 01 00
Dashboard Quick Note Set (0x1E)

TODO sent status 1e L 0 , sent size: 16 sent: 1e 10 00 29 03 01 00 01 00 03 00 01 00 01 00 00 received L len: 10,--1e 10 00 29 03 01 00 01 00 00
Pane Stock Set (0x04)

TODO cmd: L06, len: 189, data = 06 bd 00 6d 04 0b 00 01 00 00 01 02 01 01 01 08 47 4d 45 2e 58
Command 	Length 	Pad 	Sequence
06 	XX 	00 	00 ~ FF
Subcommand 	Chunk Count 	Pad 	Chunk 	Pad 			
04 	01 ~ FF 	00 	01 ~ FF 	00 			
Pane News Set (0x05)

TODO len: 115, data = 06 73 00 52 05 01 00 01 00 00 02 02 01 01 01 08 ASCII
Command 	Length 	Pad 	Sequence
06 	XX 	00 	00 ~ FF
Subcommand 	Chunk Count 	Pad 	Chunk 	Pad 			
05 	01 ~ FF 	00 	01 ~ FF 	00 			
File Upload (0x15)

TODO Image transmission currently supports 1-bit, 576*136 pixel BMP images (refer to image_1.bmp, image_2.bmp in the project). The core process includes three steps:

    Divide the BMP image data into packets (each packet is 194 bytes), then add 0x15 command and syncID to the front of the packet, and send it to the dual BLE in the order of the packets (the left and right sides can be sent independently at the same time). The first packet needs to insert 4 bytes of glasses end storage address 0x00, 0x1c, 0x00, 0x00, so the first packet data is ([0x15, index & 0xff, 0x00, 0x1c, 0x00, 0x00], pack), and other packets do not need addresses 0x00, 0x1c, 0x00, 0x00;
    After sending the last packet, it is necessary to send the packet end command [0x20, 0x0d, 0x0e] to the dual BLE;
    After the packet end command in step 2 is correctly replied, send the CRC check command to the dual BLE through the 0x16 command. When calculating the CRC, it is necessary to consider the glasses end storage address added when sending the first BMP packet.

For a specific example, click the icon in the upper right corner of the App homepage to enter the Features page. The page contains three buttons: BMP 1, BMP 2, and Exit, which represent the transmission and display of picture 1, the transmission and display of picture 2, and the exit of picture transmission and display.
Command Information

    Command: 0x15
    seq (Sequence Number): 0~255
    address: [0x00, 0x1c, 0x00, 0x00]
    data0 ~ data194

Field Descriptions

    seq (Sequence Number):
        Range: 0~255
        Description: Indicates the sequence of the current package.
    address: bmp address in the Glasses (just attached in the first pack)
    data0 ~ data194:
        bmp data packet

File Upload Complete (0x20)

Also used to upgrade a font, it seems. TODO
Pages 1

    Even Realities G1 BLE Protocol
        Operations
        Display Image Full Screen
        Display Image On Dashboard
        Display Text Full Screen
        Display Text On Dashboard
        AI Dictation and Response
        Create and Display Quick Note
        Navigation Full Screen
        Notifications
        Record Audio
        User Defined Action on Double Tap
        Scroll Text Full Screen
        Display Calendar on Dashboard
        Display Graph on Dashboard
        Messages
        Status (0x22)
        Audio (0xF1)
        Debug (0xF4)
        Event (0xF5)
        Commands
        Generic Command Response
        Anti Shake Get (0x2A)
        This is not used by the stock app. TODO
        Bitmap Show (0x16)
        Bitmap Hide (0x18)
        Brightness Set (0x01)
        Brightness Get (0x29)
        Response
        Dashboard Set (0x06)
        Response
        Time and Weather Set (0x01)
        Weather Set (0x02)
        Weather Icon IDs
        Pane Mode Set (0x06)
        Dashboard Mode IDs
        Secondary Pane IDs
        Pane Calendar Set (0x03)
        Pane Stock Set (0x04)
        Pane News Set (0x05)
        Pane Map Set (0x07)
        Send Arrow Data
        Dashboard Calendar Next Up Set (0x58)
        Dashboard Quick Note Set (0x1E)
        TODO
        File Upload (0x15)
        File Upload Complete (0x20)
        Hardware Set (0x26)
        Response
        Subcommands
        Height and Depth (0x02)
        Button Double Tap Settings (0x06)
        Actions
        Button Long Press Settings (0x07)
        Actions
        Activate Microphone on Head Lift Settings (0x08)
        Hardware Get (0x3F)
        Hardware Display Get (0x3B)
        Response
        Head Up Angle Set (0x0B)
        Head Up Action Set (0x08)
        Modes
        Head Up Angle Get (0x32)
        Response
        Head Up Calibration Control (0x10)
        Subcommands
        Info Battery and Firmware Get(0x2C)
        Response
        Info MAC Address Get (0x2D)
        Info Serial Number Lens Get (0x33)
        Response
        Info Serial NumberGlasses Get (0x34)
        Response
        Info ESB Channel Get (0x35)
        Response
        Info ESB Channel Notification Count Get (0x36)
        Response
        Info Time Since Boot Get (0x37)
        Response
        Info Buried Point Get (0x3E)
        Response
        Language Set (0x3D)
        Language ID
        Response
        TODO
        Silent Mode Set (0x03)
        Silent Mode Get (0x2B)
        Response
        State Code
        Microphone Set (0x0E)
        MTU Set (0x4D)
        Navigation Control (0x0A)
        Subcommand
        Notification App List Set (0x04)
        Notification App List Get (0x2E)
        Notification Apple Specific Get (0x38)
        Notification Auto Display Set (0x4F)
        Notification Auto Display Get (0x3C)
        Response
        Notification Send (0x4B)
        Notification Clear (0x4C)
        Status Get (0x22)
        Status Running App Get (0x39)
        System Control (0x23)
        Debug Logging Set (0x23 0x6C)
        Reboot (0x23 0x72)
        Firmware Build Info Get (0x23 0x74)
        Response
        Teleprompter Control (0x09)
        Subcommand
        Teleprompter Suspend (0x24)
        Teleprompter Position Set(0x25)
        Text Set (0x4E)
        Display Style
        Canvas State
        Timer Control (0x07)
        Transcribe Control (0x0D)
        Translate Control (0x0F)
        Tutorial Control (0x1F)
        Unpair (0x47)
        Upgrade Control (0x17)
        Wear Detection Set (0x27)
        Wear Detection Get (0x3A)
        Response
        ?? (0x50)
        Response
        Scratch
        Pane Map Set (0x07)
        Send Arrow Data
        Head Up Calibration Control (0x10)
        Status Running App Get (0x39)
        Dashboard Quick Note Set (0x1E)
        Pane Stock Set (0x04)
        Pane News Set (0x05)
        File Upload (0x15)
        Command Information
        Field Descriptions
        File Upload Complete (0x20)

Clone this wiki locally
Footer
© 2025 GitHub, Inc.
Footer navigation

    Terms
    Privacy
    Security
    Status
    Community
    Docs
    Contact

Even Realities G1 BLE Protocol · JohnRThomas/EvenDemoApp Wiki · GitHub
