// =============================================================================
//  XIAO ESP32S3 SENSE / XIAO ML KIT  -  INT8 VISION INFERENCE  -  firmware-v001
//
//  Train a tiny CNN on the web page (float), let the page convert it to INT8,
//  and run it here. This sketch never trains: it collects images and classifies.
//
//  By Jeremy Ellis   (MIT license, use at your own risk)
//  With free tier assistance from Claude (code overview), ChatGPT (critique),
//  Gemini (research) and Copilot (alternate).
//  GitHub   https://github.com/hpssjellis
//  LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// -----------------------------------------------------------------------------
//  THE MENU  (Serial monitor at 115200 baud and the 72x40 OLED)
//    1..NUM_CLASSES   collect images for that class (one folder per class)
//    NUM_CLASSES+1    Infer  (live camera classification with the INT8 model)
//
//  CONTROLS
//    Touch pad on pin A0:   1-2 taps = next item / capture
//                           3+ taps  = select / exit back to the menu
//    Serial monitor keys:   t = next / capture      l = select / exit
//                           1..9 = jump straight to a menu item (menu only)
//
// -----------------------------------------------------------------------------
//  THE SD CARD
//    /images/<class name>/img_*.jpg    240x240 photos, one folder per class
//    /header/myWeightsInt8.bin         INT8 model written by the web page (loaded at boot)
//    /header/myWeightsInt8.h           same bytes as a C array (for baking into the sketch)
//    /header/config.json               class names and "grayscale" true/false, written by the web page
//
//  WORKFLOW
//    Collect images here (or on the page), move the SD card to the computer, open
//    the Vision CNN INT8 SD trainer page, train, Convert to INT8, Save INT8 files,
//    put the card back, reboot, choose Infer.
//
// -----------------------------------------------------------------------------
//  SETTINGS THAT MUST MATCH THE WEB PAGE  (fixed when the sketch compiles)
//    INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES and myClassLabels[]
//  The page shows the exact lines to paste (section 2 "Classes and data").
//  Grayscale is NOT compiled in: it comes from "grayscale" in config.json.
//  If anything differs the device prints "REFUSED INT8 weights" with both sets of numbers.
//
//  WEIGHTS PRIORITY:  SD card myWeightsInt8.bin  >  baked-in myWeightsInt8.h
//  To bake weights in: copy /header/myWeightsInt8.h next to this sketch, set the
//  settings above to match, then remove the // in front of  #define USE_BAKED_WEIGHTS
//
//  SELF TEST: the model file holds one test image and its expected INT8 output.
//  At boot the device runs it and prints PASS or FAIL.
//
// -----------------------------------------------------------------------------
//  WEB PAGE DEBUG FRAMES
//  Nothing extra is printed unless the web page sends 'D' (it repeats it every
//  5 s; the device stops after 15 s of silence, or when it receives 'd').
//  While on, the device prints "@F ..." lines holding the camera JPEG (base64),
//  the class probabilities and a heatmap, so the page can compare its own INT8
//  model with the device on exactly the same picture.
//
// -----------------------------------------------------------------------------
//  ARDUINO IDE SETUP
//    Board: XIAO_ESP32S3     PSRAM: OPI PSRAM     Library: U8g2 (olikraus)
// =============================================================================


// Remove the // below AFTER copying myWeightsInt8.h from the SD card into the sketch folder.
//#define USE_BAKED_WEIGHTS

#ifdef USE_BAKED_WEIGHTS
  #include "myWeightsInt8.h"
#endif

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <U8g2lib.h>
#include <Wire.h>
#include "mbedtls/base64.h"


// =============================================================================
//  SETTINGS YOU MAY CHANGE
// =============================================================================

// ---- Classes -----------------------------------------------------------------
#define NUM_CLASSES 3
String myClassLabels[NUM_CLASSES] = {"0Blank", "1Cup", "2Pen"};

// ---- Model layout (must match the web page, see the header comment) -----------
// INPUT_SIZE: square and EVEN. Photos on the SD card are always 240x240 and are
// shrunk to INPUT_SIZE. Cost grows with INPUT_SIZE squared.
#define INPUT_SIZE    64
#define CONV1_FILTERS 4
#define CONV2_FILTERS 8

// ---- Touch pad (A0) -----------------------------------------------------------
// Readings above myThresholdPress count as a touch; below myThresholdRelease as released.
const int myThresholdPress   = 1100;
const int myThresholdRelease = 900;

// ---- Camera picture settings --------------------------------------------------
// The web page shows images upright and mirrored left-right. To match it, the sensor is
// mirrored AND flipped vertically. Brightness and AE level go from -2 to 2 (0 = default).
// If pictures look darker than on the web page raise them; if they wash out, lower them.
// Photos already on the SD card are not changed by these settings.
#define MY_CAM_HMIRROR        1
#define MY_CAM_VFLIP          1
#define MY_CAM_BRIGHTNESS     1
#define MY_CAM_AE_LEVEL       1
#define MY_CAM_WARMUP_FRAMES  5     // frames thrown away at start so auto exposure can settle


// =============================================================================
//  DERIVED SIZES  (calculated from the settings above, do not edit)
// =============================================================================
#define CONV1_OUTPUT_SIZE (INPUT_SIZE - 2)
#define POOL1_OUTPUT_SIZE (CONV1_OUTPUT_SIZE / 2)
#define CONV2_OUTPUT_SIZE (POOL1_OUTPUT_SIZE - 2)
#define FLATTENED_SIZE    (CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS)

#define CONV1_OUT_BYTES   (CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * CONV1_FILTERS)
#define POOL1_OUT_BYTES   (POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * CONV1_FILTERS)

static_assert(INPUT_SIZE % 2 == 0, "INPUT_SIZE must be even");
static_assert(CONV2_OUTPUT_SIZE >= 1, "INPUT_SIZE is too small");
static_assert(NUM_CLASSES >= 2, "Need at least 2 classes");

// =============================================================================
//  XIAO ESP32S3 CAMERA PINS
// =============================================================================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13


// =============================================================================
//  GLOBAL VARIABLES  (all declared here, above every function)
// =============================================================================

// ---- Display ------------------------------------------------------------------
U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ---- Hardware status ----------------------------------------------------------
bool mySDavailable = false;       // true when the SD card mounted
bool myCameraOk    = false;       // true when the camera started
bool myModelLoaded = false;       // true once an INT8 model is in memory

// ---- Input channels (from config.json, 3 = RGB, 1 = grayscale) ----------------
bool myConfigHasGray = false;
int  myChannels = 3;

// ---- Menu ---------------------------------------------------------------------
const int myTotalItems = NUM_CLASSES + 1;   // one per class, then Infer
int  myMenuIndex = 1;
bool myIsSelected = false;
unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime = 0;
const int myTapCooldown = 250;

// ---- Touch input: counts taps inside a short time window ----------------------
struct TouchState {
  bool isTouching = false;
  int tapCount = 0;
  unsigned long firstTapTime = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime = 0;
};
TouchState myTouch;
const unsigned long myTapWindow = 800;     // ms to wait for more taps
const int           myLongPressTaps = 3;   // this many taps counts as a "long press" (select / exit)
const unsigned long myDebounceMs = 50;

// ---- Image buffers ------------------------------------------------------------
uint8_t* myRgbBuffer = nullptr;            // one decoded 240x240 RGB picture
int myResizeLut[INPUT_SIZE];               // source pixel for each model pixel (nearest neighbour)

// ---- INT8 model (PSRAM) -------------------------------------------------------
// Each layer: int8 weights, int32 bias (input zero point already folded in),
// requantize multiplier + shift for positive and negative sums, output zero point.
struct MyQLayer {
  int8_t*  w;
  int32_t* bias;
  int32_t* mp;
  int32_t* sp;
  int32_t* mn;
  int32_t* sn;
  int32_t  zp;
};
MyQLayer myL1;
MyQLayer myL2;
MyQLayer myLD;
float   myLogitScale = 1.0f;
int32_t myLogitZp = 0;

// ---- INT8 buffers -------------------------------------------------------------
int8_t* myIn = nullptr;                    // model input, pixel - 128
int8_t* myC1 = nullptr;
int8_t* myP1 = nullptr;
int8_t* myC2 = nullptr;
int8_t  myLogitsQ[NUM_CLASSES];
float   myProbs[NUM_CLASSES];

// ---- Self test data from the model file ---------------------------------------
uint8_t* myTestImage = nullptr;
int8_t   myTestLogits[NUM_CLASSES];
bool     myHasTest = false;

// ---- Web page debug frames ----------------------------------------------------
bool myDebugStream = false;
unsigned long myDebugLastSeen = 0;


// =============================================================================
//  FORWARD DECLARATIONS  (every function, so the order below does not matter)
// =============================================================================

// math helpers
inline int32_t myRq(int32_t acc, int32_t m, int32_t n);
inline int8_t  myClamp8(int32_t v);

// OLED
void myShowMessage(const char* l1, const char* l2 = nullptr, const char* l3 = nullptr);
void myRenderRgbToOLED(int imageCount);
void myDisplayImageOnOLED(camera_fb_t* fb, int imageCount);

// touch
int  myReadTouch();
void myResetTouchState();
void myUpdateTouchState();
int  myCheckTouchInput();
void myCheckTouchBackground();
int  myPeekTouchAction();

// memory and model file
void   myAllocateMemory();
bool   myTake(const uint8_t*& p, const uint8_t* end, void* dst, size_t len);
bool   myTakeLayer(const uint8_t*& p, const uint8_t* end, MyQLayer& L, int wCount, int count);
size_t myExpectedBlobBytes();
bool   myParseBlob(const uint8_t* blob, size_t n);
bool   myLoadWeights();
void   mySelfTest();

// config.json
static bool myJsonInt(const String& t, const char* key, int& out);
static bool myJsonBool(const String& t, const char* key, bool& out);
static int  myJsonStringArray(const String& t, const char* key, std::vector<String>& out);
void myLoadConfig();

// debug frames
bool myHandleDebugChar(char c);
static void myPrintB64(const uint8_t* d, size_t n);
void myDebugSendFrame(char kind, int n, camera_fb_t* fb, int pred, const float* probs);

// images
void   myResizeToInput(const uint8_t* rgb240, int8_t* dst);
String myUniqueImagePath(const String& folder);

// the CNN
void myForwardInt8();

// menu actions
void myActionCollect(int classIdx);
void myActionInfer();

// menu
String myMenuLabel(int item);
void myResetMenuState();
void myDrawMenu();
void myExecuteMenuItem(int idx);
void myHandleMenuNavigation();


// =============================================================================
//  SETUP AND LOOP
// =============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(1000);

  Serial.println("\n=== XIAO ESP32S3 INT8 Vision (firmware-v001) ===");
  Serial.printf("Layout: INPUT_SIZE %d, CONV1_FILTERS %d, CONV2_FILTERS %d, NUM_CLASSES %d\n",
                INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES);
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

  myRgbBuffer = (uint8_t*)ps_malloc(240 * 240 * 3);
  if (!myRgbBuffer) {
    Serial.println("FATAL: could not allocate the picture buffer. Is PSRAM set to OPI PSRAM?");
    myShowMessage("PSRAM ERROR!", "Set PSRAM to", "OPI PSRAM");
    while (1) { delay(1000); }
  }

  // Pre-compute which source pixel feeds each model pixel (240x240 -> INPUT_SIZE).
  for (int i = 0; i < INPUT_SIZE; i++) {
    myResizeLut[i] = min((int)((i + 0.5) * 240.0 / INPUT_SIZE), 239);
  }

  pinMode(A0, INPUT);
  u8g2.begin();
  u8g2.setFont(u8g2_font_6x10_tf);

  // ---- SD card (started slowly with a timeout so a missing card cannot hang the boot) ----
  pinMode(21, OUTPUT);
  digitalWrite(21, HIGH);
  delay(100);
  Serial.println("Checking SD card...");
  SPI.begin();
  SPI.setFrequency(400000);
  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);

  if (!mySDavailable) {
    SD.end();
    Serial.println("No SD card - continuing without it");
    myShowMessage("No SD card");
    delay(2000);
  } else {
    Serial.println("SD card mounted successfully");
    myLoadConfig();      // class names from /header/config.json
  }

  // ---- Camera ----
  camera_config_t config = {};     // all fields start at zero (= library defaults)
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_240X240;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  esp_err_t camErr = esp_camera_init(&config);
  if (camErr != ESP_OK) {
    Serial.printf("Camera init FAILED: 0x%x\n", camErr);
  } else {
    myCameraOk = true;
    Serial.println("Camera initialized");
    sensor_t* s = esp_camera_sensor_get();
    if (s != NULL) {
      s->set_hmirror(s, MY_CAM_HMIRROR);
      s->set_vflip(s, MY_CAM_VFLIP);
      s->set_brightness(s, MY_CAM_BRIGHTNESS);
      s->set_ae_level(s, MY_CAM_AE_LEVEL);
    }
    // Throw away the first frames so auto exposure settles before any image is used.
    for (int i = 0; i < MY_CAM_WARMUP_FRAMES; i++) {
      camera_fb_t* warm = esp_camera_fb_get();
      if (warm) esp_camera_fb_return(warm);
      delay(60);
    }
  }

  // Quieter serial output: only warnings and errors from the ESP-IDF libraries.
  esp_log_level_set("*", ESP_LOG_WARN);
  esp_log_level_set("esp_camera", ESP_LOG_ERROR);

  myAllocateMemory();

#ifdef USE_BAKED_WEIGHTS
  if (!myConfigHasGray) myChannels = myInt8Blob[8];
  if (myParseBlob(myInt8Blob, sizeof(myInt8Blob))) Serial.println("Baked-in INT8 weights loaded from myWeightsInt8.h");
#endif

  if (myLoadWeights()) {
    Serial.println("SD INT8 weights loaded");
  }

  Serial.printf("Input channels: %d (%s)\n", myChannels, myChannels == 1 ? "grayscale" : "RGB");
  mySelfTest();

  myLastActivityTime = millis();
  myResetMenuState();
  delay(2000);             // time to open the serial monitor

  Serial.println("System ready - Tap A0 to navigate, 3+ taps to select");
  myDrawMenu();
}

void loop() {
  myHandleMenuNavigation();
}


// =============================================================================
//  MATH HELPERS
//  The INT8 math is exactly the same as in the web page, so the results match bit for bit.
// =============================================================================

// (acc * m) / 2^n rounded, using a 64-bit product.
inline int32_t myRq(int32_t acc, int32_t m, int32_t n) {
  int64_t p = (int64_t)acc * m + ((int64_t)1 << (n - 1));
  return (int32_t)(p >> n);
}

inline int8_t myClamp8(int32_t v) {
  return (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
}


// =============================================================================
//  OLED HELPERS
// =============================================================================

// Up to three short lines (about 14 characters each fit on the 72x40 display).
void myShowMessage(const char* l1, const char* l2, const char* l3) {
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    if (l1) u8g2.drawStr(0, 10, l1);
    if (l2) u8g2.drawStr(0, 22, l2);
    if (l3) u8g2.drawStr(0, 34, l3);
  } while (u8g2.nextPage());
}

// Draws myRgbBuffer (must already hold a 240x240 picture) as a black and white image.
// imageCount >= 0 shows that number in the corner (after a capture); -1 shows "LIVE".
void myRenderRgbToOLED(int imageCount) {
  int myOledWidth  = u8g2.getDisplayWidth();
  int myOledHeight = u8g2.getDisplayHeight();
  int myScaleX = 240 / myOledWidth;
  int myScaleY = 240 / myOledHeight;

  u8g2.firstPage();
  do {
    for (int myOledX = 0; myOledX < myOledWidth; myOledX++) {
      for (int myOledY = 0; myOledY < myOledHeight; myOledY++) {
        size_t myPixelIndex = ((myOledY * myScaleY) * 240 + (myOledX * myScaleX)) * 3;
        uint8_t myBrightness = (myRgbBuffer[myPixelIndex]     +
                                myRgbBuffer[myPixelIndex + 1] +
                                myRgbBuffer[myPixelIndex + 2]) / 3;
        if (myBrightness > 100) u8g2.drawPixel(myOledX, myOledY);
      }
    }
    if (imageCount >= 0) {
      u8g2.setFont(u8g2_font_ncenB10_tr);
      u8g2.setColorIndex(0);
      u8g2.drawBox(0, 0, 20, 15);
      u8g2.setColorIndex(1);
      u8g2.setCursor(3, 10);
      u8g2.print(String(imageCount));
    } else {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.setColorIndex(0);
      u8g2.drawBox(50, 0, 22, 8);
      u8g2.setColorIndex(1);
      u8g2.drawStr(52, 7, "LIVE");
    }
  } while (u8g2.nextPage());
}

// Decode a camera JPEG into myRgbBuffer and show it with a count badge.
void myDisplayImageOnOLED(camera_fb_t* fb, int imageCount) {
  if (!myRgbBuffer) return;
  if (!fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) {
    Serial.println("Failed to convert JPEG to RGB888 for OLED");
    return;
  }
  myRenderRgbToOLED(imageCount);
}


// =============================================================================
//  TOUCH INPUT
//  A tap is a touch and release. Taps inside myTapWindow are counted together:
//    1-2 taps -> "tap"  (returns 1)      3 or more taps -> "long press" (returns 2)
//  Call myCheckTouchInput() from a loop to receive the result once the window ends.
// =============================================================================

// Average of three readings of the touch pad.
int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) {
    sum += analogRead(A0);
    delayMicroseconds(100);
  }
  return sum / 3;
}

void myResetTouchState() {
  myTouch.isTouching = false;
  myTouch.tapCount = 0;
  myTouch.firstTapTime = 0;
  myTouch.lastReleaseTime = 0;
  myTouch.lastCheckTime = 0;
}

// Reads the pad (at most every 20 ms) and counts taps.
void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;

  int val = myReadTouch();
  bool touchActive = myTouch.isTouching ? (val > myThresholdRelease) : (val > myThresholdPress);

  // A new touch just started.
  if (touchActive && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myDebounceMs) return;
    myTouch.isTouching = true;

    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
      Serial.printf("Tap #%d\n", myTouch.tapCount);
    } else {
      myTouch.tapCount = 1;               // the window ran out: start counting again
      myTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }

  // The touch ended.
  if (!touchActive && myTouch.isTouching) {
    myTouch.isTouching = false;
    myTouch.lastReleaseTime = now;
  }
}

// Returns 0 = nothing yet, 1 = tap, 2 = long press (3+ taps). Consumes the result.
int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();

  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTapWindow) {
      int result = (myTouch.tapCount >= myLongPressTaps) ? 2 : 1;
      int count = myTouch.tapCount;
      myResetTouchState();
      if (result == 2) Serial.printf("LONG PRESS detected (%d taps)\n", count);
      else             Serial.printf("TAP detected (%d tap%s)\n", count, count > 1 ? "s" : "");
      return result;
    }
  }
  return 0;
}

// Keeps the tap counter up to date without consuming a result (use inside heavy loops).
void myCheckTouchBackground() {
  myUpdateTouchState();
}

// Same answer as myCheckTouchInput() but does not consume it.
int myPeekTouchAction() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTapWindow) {
      return (myTouch.tapCount >= myLongPressTaps) ? 2 : 1;
    }
  }
  return 0;
}




// =============================================================================
//  MEMORY AND THE INT8 MODEL FILE
//  File layout (little endian):
//    32 byte header: "WI81", version u16, input size u16, channels u8, conv1 filters u8,
//                    conv2 filters u8, classes u8, logit scale float, logit zero point int32
//    then for conv1, conv2 and dense:  int8 weights, int32 bias, multiplier positive,
//                    shift positive, multiplier negative, shift negative (one each per
//                    filter or class), int32 output zero point
//    then the test image (uint8 pixels) and the expected INT8 logits
// =============================================================================
void myAllocateMemory() {
  if (myIn != nullptr) return;

  Serial.println("\n=== Allocating inference memory ===");

  myIn = (int8_t*)ps_malloc(INPUT_SIZE * INPUT_SIZE * 3);
  myC1 = (int8_t*)ps_malloc(CONV1_OUT_BYTES);
  myP1 = (int8_t*)ps_malloc(POOL1_OUT_BYTES);
  myC2 = (int8_t*)ps_malloc(FLATTENED_SIZE);
  myTestImage = (uint8_t*)ps_malloc(INPUT_SIZE * INPUT_SIZE * 3);

  myL1.w = (int8_t*)ps_malloc(CONV1_FILTERS * 27);
  myL2.w = (int8_t*)ps_malloc(CONV2_FILTERS * 9 * CONV1_FILTERS);
  myLD.w = (int8_t*)ps_malloc((size_t)FLATTENED_SIZE * NUM_CLASSES);

  MyQLayer* layers[3] = { &myL1, &myL2, &myLD };
  int counts[3] = { CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES };
  bool ok = myIn && myC1 && myP1 && myC2 && myTestImage && myL1.w && myL2.w && myLD.w;
  for (int i = 0; i < 3; i++) {
    layers[i]->bias = (int32_t*)ps_malloc(counts[i] * 4);
    layers[i]->mp   = (int32_t*)ps_malloc(counts[i] * 4);
    layers[i]->sp   = (int32_t*)ps_malloc(counts[i] * 4);
    layers[i]->mn   = (int32_t*)ps_malloc(counts[i] * 4);
    layers[i]->sn   = (int32_t*)ps_malloc(counts[i] * 4);
    layers[i]->zp   = 0;
    ok = ok && layers[i]->bias && layers[i]->mp && layers[i]->sp && layers[i]->mn && layers[i]->sn;
  }

  if (!ok) {
    Serial.println("FATAL: PSRAM allocation failed! Even inference does not fit.");
    Serial.println("Use a smaller INPUT_SIZE, or check that PSRAM is set to OPI PSRAM.");
    myShowMessage("PSRAM ERROR!", "Model too big", "or no PSRAM");
    while (1) { delay(1000); }
  }

  Serial.printf("Free PSRAM after inference allocation: %d bytes\n", ESP.getFreePsram());
}

// Copies len bytes from the file data and moves the read position. False if the data ran out.
bool myTake(const uint8_t*& p, const uint8_t* end, void* dst, size_t len) {
  if ((size_t)(end - p) < len) return false;
  memcpy(dst, p, len);
  p += len;
  return true;
}

bool myTakeLayer(const uint8_t*& p, const uint8_t* end, MyQLayer& L, int wCount, int count) {
  return myTake(p, end, L.w, wCount) &&
         myTake(p, end, L.bias, count * 4) &&
         myTake(p, end, L.mp, count * 4) &&
         myTake(p, end, L.sp, count * 4) &&
         myTake(p, end, L.mn, count * 4) &&
         myTake(p, end, L.sn, count * 4) &&
         myTake(p, end, &L.zp, 4);
}

// Exact size in bytes of myWeightsInt8.bin for this layout and channel count.
size_t myExpectedBlobBytes() {
  size_t n = 32;
  n += (size_t)CONV1_FILTERS * 9 * myChannels + CONV1_FILTERS * 20 + 4;
  n += (size_t)CONV2_FILTERS * 9 * CONV1_FILTERS + CONV2_FILTERS * 20 + 4;
  n += (size_t)FLATTENED_SIZE * NUM_CLASSES + NUM_CLASSES * 20 + 4;
  n += (size_t)INPUT_SIZE * INPUT_SIZE * myChannels + NUM_CLASSES;
  return n;
}

// Checks the header against this sketch and loads the model. Refuses on any difference.
bool myParseBlob(const uint8_t* b, size_t n) {
  if (n < 32 || memcmp(b, "WI81", 4) != 0) {
    Serial.println("REFUSED INT8 weights: this is not a WI81 model file");
    return false;
  }
  uint16_t ver, inSize;
  memcpy(&ver, b + 4, 2);
  memcpy(&inSize, b + 6, 2);
  int ch = b[8], c1 = b[9], c2 = b[10], nc = b[11];
  float ls;
  int32_t lz;
  memcpy(&ls, b + 12, 4);
  memcpy(&lz, b + 16, 4);

  if (ver != 1 || inSize != INPUT_SIZE || ch != myChannels || c1 != CONV1_FILTERS || c2 != CONV2_FILTERS || nc != NUM_CLASSES) {
    Serial.printf("REFUSED INT8 weights: file has input %u, channels %d, filters %d/%d, classes %d\n", (unsigned)inSize, ch, c1, c2, nc);
    Serial.printf("Sketch needs input %d, channels %d (grayscale %s in config.json), filters %d/%d, classes %d\n",
                  INPUT_SIZE, myChannels, myChannels == 1 ? "true" : "false", CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES);
    Serial.println("Make the settings match the web page, reflash, or convert again.");
    return false;
  }
  if (n != myExpectedBlobBytes()) {
    Serial.printf("REFUSED INT8 weights: file is %u bytes but this sketch needs %u bytes\n", (unsigned)n, (unsigned)myExpectedBlobBytes());
    return false;
  }

  const uint8_t* p = b + 32;
  const uint8_t* end = b + n;
  bool ok = myTakeLayer(p, end, myL1, CONV1_FILTERS * 9 * ch, CONV1_FILTERS) &&
            myTakeLayer(p, end, myL2, CONV2_FILTERS * 9 * CONV1_FILTERS, CONV2_FILTERS) &&
            myTakeLayer(p, end, myLD, FLATTENED_SIZE * NUM_CLASSES, NUM_CLASSES) &&
            myTake(p, end, myTestImage, INPUT_SIZE * INPUT_SIZE * ch) &&
            myTake(p, end, myTestLogits, NUM_CLASSES);
  if (!ok) {
    Serial.println("REFUSED INT8 weights: the file ended early");
    return false;
  }
  myLogitScale = ls;
  myLogitZp = lz;
  myHasTest = true;
  myModelLoaded = true;
  return true;
}

// Loads /header/myWeightsInt8.bin from the SD card.
bool myLoadWeights() {
  if (!mySDavailable) {
    Serial.println("No SD card - skipping weight load");
    return false;
  }
  if (!SD.exists("/header/myWeightsInt8.bin")) {
    Serial.println("No SD INT8 weights file found");
    return false;
  }
  Serial.println("Loading INT8 weights from SD...");
  File f = SD.open("/header/myWeightsInt8.bin", FILE_READ);
  if (!f) return false;
  size_t sz = f.size();
  uint8_t* buf = (uint8_t*)ps_malloc(sz);
  if (!buf) {
    Serial.println("Not enough PSRAM to read the weights file");
    f.close();
    return false;
  }
  f.read(buf, sz);
  f.close();
  bool ok = myParseBlob(buf, sz);
  free(buf);
  return ok;
}

// Runs the test image stored in the model file and compares with the expected INT8 logits.
void mySelfTest() {
  if (!myHasTest || !myModelLoaded) return;
  int total = INPUT_SIZE * INPUT_SIZE * myChannels;
  for (int i = 0; i < total; i++) myIn[i] = (int8_t)((int)myTestImage[i] - 128);
  myForwardInt8();
  int maxDiff = 0;
  for (int c = 0; c < NUM_CLASSES; c++) {
    int d = abs((int)myLogitsQ[c] - (int)myTestLogits[c]);
    if (d > maxDiff) maxDiff = d;
  }
  if (maxDiff == 0) {
    Serial.println("INT8 self-test PASS: the device matches the web page exactly");
  } else {
    Serial.printf("INT8 self-test FAIL: largest logit difference %d\n", maxDiff);
    Serial.print("  device: ");
    for (int c = 0; c < NUM_CLASSES; c++) Serial.printf("%d ", (int)myLogitsQ[c]);
    Serial.print("\n  page:   ");
    for (int c = 0; c < NUM_CLASSES; c++) Serial.printf("%d ", (int)myTestLogits[c]);
    Serial.println();
  }
}


// Finds  "key": 123  in the text. Returns true and sets out when found.
static bool myJsonInt(const String& t, const char* key, int& out) {
  String k("\"");
  k += key;
  k += "\"";
  int p = t.indexOf(k.c_str());
  if (p < 0) return false;
  p = t.indexOf(':', p);
  if (p < 0) return false;
  p++;
  int len = (int)t.length();
  while (p < len && (t[p] == ' ' || t[p] == '\n' || t[p] == '\r' || t[p] == '\t')) p++;
  bool neg = false;
  if (p < len && t[p] == '-') { neg = true; p++; }
  if (p >= len || t[p] < '0' || t[p] > '9') return false;
  long v = 0;
  while (p < len && t[p] >= '0' && t[p] <= '9') { v = v * 10 + (t[p] - '0'); p++; }
  out = (int)(neg ? -v : v);
  return true;
}

// Reads the list of strings named key. Returns how many were found, or -1 if the key or list is missing.
static int myJsonStringArray(const String& t, const char* key, std::vector<String>& out) {
  String k("\"");
  k += key;
  k += "\"";
  int p = t.indexOf(k.c_str());
  if (p < 0) return -1;
  p = t.indexOf('[', p);
  if (p < 0) return -1;
  p++;
  int len = (int)t.length();
  out.clear();
  while (p < len) {
    char c = t[p];
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == ',') { p++; continue; }
    if (c == ']') break;
    if (c != '"') return -1;                  // not a list of strings
    p++;
    String s("");
    while (p < len && t[p] != '"') {
      if (t[p] == '\\' && p + 1 < len) p++;   // keep the escaped character
      s += t[p];
      p++;
    }
    if (p >= len) return -1;                  // string never closed
    p++;                                      // closing quote
    out.push_back(s);
  }
  return (int)out.size();
}

// Finds  "key": true  or  "key": false  in the text. Returns true when found.
static bool myJsonBool(const String& t, const char* key, bool& out) {
  String k("\"");
  k += key;
  k += "\"";
  int p = t.indexOf(k.c_str());
  if (p < 0) return false;
  p = t.indexOf(':', p);
  if (p < 0) return false;
  p++;
  int len = (int)t.length();
  while (p < len && (t[p] == ' ' || t[p] == '\n' || t[p] == '\r' || t[p] == '\t')) p++;
  if (t.startsWith("true", p)) { out = true; return true; }
  if (t.startsWith("false", p)) { out = false; return true; }
  return false;
}

void myLoadConfig() {
  if (!mySDavailable) return;
  if (!SD.exists("/header/config.json")) {
    Serial.println("No /header/config.json - using compiled class labels and RGB input");
    return;
  }
  File f = SD.open("/header/config.json", FILE_READ);
  if (!f) return;
  size_t sz = f.size();
  if (sz == 0 || sz > 4096) {
    Serial.printf("config.json ignored: unexpected size %u bytes\n", (unsigned)sz);
    f.close();
    return;
  }
  String txt;
  txt.reserve(sz + 1);
  while (f.available()) txt += (char)f.read();
  f.close();

  std::vector<String> names;
  int n = myJsonStringArray(txt, "classes", names);
  if (n < 0) {
    Serial.println("config.json has no readable \"classes\" list - keeping the compiled class labels");
  } else if (n != NUM_CLASSES) {
    Serial.printf("config.json lists %d classes but the sketch has NUM_CLASSES %d - keeping the compiled labels.\n", n, NUM_CLASSES);
  } else {
    Serial.println("Class labels loaded from /header/config.json:");
    for (int i = 0; i < NUM_CLASSES; i++) {
      myClassLabels[i] = names[i];
      Serial.printf("  %d: %s\n", i, myClassLabels[i].c_str());
    }
  }

  bool gray;
  if (myJsonBool(txt, "grayscale", gray)) {
    myConfigHasGray = true;
    myChannels = gray ? 1 : 3;
  }

  int v;
  if (myJsonInt(txt, "input_size", v) && v != INPUT_SIZE)
    Serial.printf("WARNING: config.json input_size %d but the sketch INPUT_SIZE is %d\n", v, INPUT_SIZE);
  if (myJsonInt(txt, "conv1_filters", v) && v != CONV1_FILTERS)
    Serial.printf("WARNING: config.json conv1_filters %d but the sketch CONV1_FILTERS is %d\n", v, CONV1_FILTERS);
  if (myJsonInt(txt, "conv2_filters", v) && v != CONV2_FILTERS)
    Serial.printf("WARNING: config.json conv2_filters %d but the sketch CONV2_FILTERS is %d\n", v, CONV2_FILTERS);
}


// =============================================================================
//  WEB PAGE DEBUG FRAMES
//    @F <kind> <n> <pred> <probs|-> <logits|-> <layout> <centre input|-> <heatSide> <heat base64|-> <jpeg base64>
//  kind I = inference frame (every 10th), C = image just saved, P = live preview while collecting.
//  The heatmap is the maximum over the conv2 filters, scaled 0..255.
// =============================================================================
// Returns true when the character was a debug command (so the caller ignores it).
bool myHandleDebugChar(char c) {
  if (c == 'D') {
    if (!myDebugStream) Serial.println("Debug frames ON");
    myDebugStream = true;
    myDebugLastSeen = millis();
    return true;
  }
  if (c == 'd') {
    if (myDebugStream) Serial.println("Debug frames OFF");
    myDebugStream = false;
    return true;
  }
  return false;
}

// Prints bytes as base64 in chunks of 384 bytes (a multiple of 3, so the pieces join cleanly).
static void myPrintB64(const uint8_t* d, size_t n) {
  unsigned char out[520];
  while (n > 0) {
    size_t take = n > 384 ? 384 : n;
    size_t ol = 0;
    mbedtls_base64_encode(out, sizeof(out), &ol, d, take);
    Serial.write(out, ol);
    d += take;
    n -= take;
  }
}


void myDebugSendFrame(char kind, int n, camera_fb_t* fb, int pred, const float* probs) {
  if (!myDebugStream || !fb || !Serial) return;
  if (millis() - myDebugLastSeen > 15000) { myDebugStream = false; return; }   // the page stopped sending
  const bool inf = (kind == 'I' && probs != nullptr);
  Serial.printf("@F %c %d %d ", kind, n, pred);
  if (inf) {
    for (int i = 0; i < NUM_CLASSES; i++) { if (i) Serial.print(','); Serial.print(probs[i], 4); }
    Serial.print(' ');
    for (int i = 0; i < NUM_CLASSES; i++) { if (i) Serial.print(','); Serial.print(((int)myLogitsQ[i] - myLogitZp) * myLogitScale, 4); }
  } else {
    Serial.print("- -");
  }
  Serial.printf(" %dx%dx%dx%d ", INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, myChannels);
  if (inf) {
    int c = ((INPUT_SIZE / 2) * INPUT_SIZE + INPUT_SIZE / 2) * myChannels;
    for (int k = 0; k < myChannels; k++) { if (k) Serial.print(','); Serial.print(((int)myIn[c + k] + 128) / 255.0f, 4); }
  } else {
    Serial.print('-');
  }
  Serial.print(' ');
  if (inf) {
    const int hn = CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    static uint8_t heat[CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE];
    float lo = 1e30f, hi = -1e30f;
    for (int i = 0; i < hn; i++) {
      float m = myC2[i];
      for (int f = 1; f < CONV2_FILTERS; f++) { float v = myC2[f * hn + i]; if (v > m) m = v; }
      if (m < lo) lo = m;
      if (m > hi) hi = m;
    }
    float span = hi - lo;
    if (span < 1e-9f) span = 1.0f;
    for (int i = 0; i < hn; i++) {
      float m = myC2[i];
      for (int f = 1; f < CONV2_FILTERS; f++) { float v = myC2[f * hn + i]; if (v > m) m = v; }
      heat[i] = (uint8_t)(255.0f * (m - lo) / span + 0.5f);
    }
    Serial.printf("%d ", CONV2_OUTPUT_SIZE);
    myPrintB64(heat, hn);
  } else {
    Serial.print("0 -");
  }
  Serial.print(' ');
  myPrintB64(fb->buf, fb->len);
  Serial.println();
}


// =============================================================================
//  IMAGES
// =============================================================================

// Shrinks the 240x240 RGB picture to INPUT_SIZE x INPUT_SIZE INT8 values (pixel - 128).
// Grayscale uses the whole-number weights 77, 150, 29 (they add to 256), the same as the web page.
void myResizeToInput(const uint8_t* rgb240, int8_t* dst) {
  for (int y = 0; y < INPUT_SIZE; y++) {
    int rowOffset = myResizeLut[y] * 240;
    for (int x = 0; x < INPUT_SIZE; x++) {
      int srcIdx = (rowOffset + myResizeLut[x]) * 3;
      if (myChannels == 1) {
        int g = (77 * rgb240[srcIdx] + 150 * rgb240[srcIdx + 1] + 29 * rgb240[srcIdx + 2]) >> 8;
        dst[y * INPUT_SIZE + x] = (int8_t)(g - 128);
      } else {
        int dstIdx = (y * INPUT_SIZE + x) * 3;
        dst[dstIdx]     = (int8_t)((int)rgb240[srcIdx]     - 128);
        dst[dstIdx + 1] = (int8_t)((int)rgb240[srcIdx + 1] - 128);
        dst[dstIdx + 2] = (int8_t)((int)rgb240[srcIdx + 2] - 128);
      }
    }
  }
}

// A file name inside folder that does not exist yet (so a new photo never overwrites an old one).
String myUniqueImagePath(const String& folder) {
  unsigned long n = millis();
  String p = folder + "/img_" + String(n) + ".jpg";
  while (SD.exists(p)) {
    n++;
    p = folder + "/img_" + String(n) + ".jpg";
  }
  return p;
}


// =============================================================================
//  MENU ITEM 1..NUM_CLASSES : COLLECT IMAGES
//  Shows a live preview on the OLED and saves 240x240 JPEGs into /images/<class>/.
//  A tap (or 't') takes a picture. 3+ taps (or 'l') returns to the menu.
// =============================================================================
void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot collect images");
    myShowMessage("No SD card");
    delay(2000);
    myResetMenuState();
    return;
  }
  if (!myCameraOk) {
    Serial.println("Camera not working - cannot collect images");
    myShowMessage("Camera error");
    delay(2000);
    myResetMenuState();
    return;
  }

  Serial.printf("\n>>> Collection mode: %s\n", myClassLabels[classIdx].c_str());
  Serial.println("  TAP (1-2 taps) = capture image      LONG PRESS (3+ taps) = exit to menu");
  Serial.println("  Serial: 't' = capture, 'l' = exit");

  myResetTouchState();

  String folder = "/images/" + myClassLabels[classIdx];
  if (!SD.exists("/images")) SD.mkdir("/images");
  if (!SD.exists(folder)) SD.mkdir(folder);

  // Count the images already in this class folder.
  int imageCount = 0;
  File root = SD.open(folder);
  if (root) {
    while (File file = root.openNextFile()) {
      String fn = String(file.name());
      if (!file.isDirectory() && (fn.endsWith(".jpg") || fn.endsWith(".JPG"))) imageCount++;
      file.close();
    }
    root.close();
  }
  Serial.printf("This class already has %d images\n", imageCount);

  unsigned long lastCameraDrain = 0;    // last time the camera buffer was serviced
  unsigned long lastOLED = 0;           // last time the OLED preview was refreshed
  unsigned long lastDebugPreview = 0;   // last time a preview went to the web page
  bool oledNeedsUpdate = false;
  bool shouldCapture = false;

  while (true) {
    unsigned long now = millis();

    // Every 50 ms take a frame out of the camera so its buffer never overflows.
    if (now - lastCameraDrain > 50) {
      lastCameraDrain = now;

      if (!shouldCapture) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          if (myDebugStream && now - lastDebugPreview > 1000) {
            lastDebugPreview = now;
            myDebugSendFrame('P', imageCount, fb, -1, nullptr);
          }
          // Decoding to RGB is slow, so only do it when the OLED is due for a refresh.
          if (now - lastOLED > 250 && myRgbBuffer) {
            if (fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) {
              oledNeedsUpdate = true;
              lastOLED = now;
            }
          }
          esp_camera_fb_return(fb);
        }
      }
    }

    if (oledNeedsUpdate) {
      oledNeedsUpdate = false;
      myRenderRgbToOLED(-1);            // -1 = LIVE badge
    }

    // Serial keys.
    if (Serial.available()) {
      char c = Serial.read();
      if (myHandleDebugChar(c)) {
        // handled: the web page is asking for debug frames
      } else if (c == 'l' || c == 'L') {
        myResetMenuState();
        return;
      } else if (c == 't' || c == 'T') {
        shouldCapture = true;
      }
    }

    // Touch pad.
    int touchAction = myCheckTouchInput();
    if (touchAction == 2) {
      Serial.println("Exiting collection mode");
      myResetMenuState();
      return;
    } else if (touchAction == 1) {
      shouldCapture = true;
    }

    // Save one picture.
    if (shouldCapture) {
      shouldCapture = false;
      camera_fb_t* fb = esp_camera_fb_get();
      if (fb) {
        String fileName = myUniqueImagePath(folder);
        File file = SD.open(fileName, FILE_WRITE);
        if (file) {
          file.write(fb->buf, fb->len);
          file.close();
          imageCount++;
          Serial.printf("Saved: %s (Total: %d)\n", fileName.c_str(), imageCount);
          myDisplayImageOnOLED(fb, imageCount);
          myDebugSendFrame('C', imageCount, fb, -1, nullptr);
          delay(300);                   // let the snapshot show before the live view returns
          lastOLED = millis();
        }
        esp_camera_fb_return(fb);
      }
    }

    delay(5);
  }
}


// =============================================================================
//  THE INT8 CNN
//    input INPUT_SIZE x INPUT_SIZE x channels (pixel - 128)
//    -> conv 3x3 (CONV1_FILTERS) -> leaky ReLU -> 2x2 max pool
//    -> conv 3x3 (CONV2_FILTERS) -> leaky ReLU
//    -> flatten -> dense (NUM_CLASSES) -> softmax
//  All sums are int32. Each layer ends with a requantize back to int8.
//  Data layout is [filter][row][column]; the input is [row][column][channel].
// =============================================================================
void myForwardInt8() {
  const int ch = myChannels;
  const int stride1 = 9 * ch;

  // Conv1
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    for (int y = 0; y < CONV1_OUTPUT_SIZE; y++) {
      for (int x = 0; x < CONV1_OUTPUT_SIZE; x++) {
        int32_t a = myL1.bias[f];
        for (int ky = 0; ky < 3; ky++) {
          for (int kx = 0; kx < 3; kx++) {
            const int8_t* ip = myIn + ((y + ky) * INPUT_SIZE + (x + kx)) * ch;
            const int8_t* wp = myL1.w + f * stride1 + (ky * 3 + kx) * ch;
            if (ch == 3) a += ip[0] * wp[0] + ip[1] * wp[1] + ip[2] * wp[2];
            else         a += ip[0] * wp[0];
          }
        }
        int32_t r = (a >= 0) ? myRq(a, myL1.mp[f], myL1.sp[f]) : myRq(a, myL1.mn[f], myL1.sn[f]);
        myC1[ob + y * CONV1_OUTPUT_SIZE + x] = myClamp8(myL1.zp + r);
      }
    }
  }

  // Pool1: keep the largest of each 2x2 block
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ib = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE, ob = f * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
    for (int y = 0; y < POOL1_OUTPUT_SIZE; y++) {
      for (int x = 0; x < POOL1_OUTPUT_SIZE; x++) {
        int i0 = ib + y * 2 * CONV1_OUTPUT_SIZE + x * 2;
        int i2 = i0 + CONV1_OUTPUT_SIZE;
        int8_t m = myC1[i0];
        if (myC1[i0 + 1] > m) m = myC1[i0 + 1];
        if (myC1[i2] > m) m = myC1[i2];
        if (myC1[i2 + 1] > m) m = myC1[i2 + 1];
        myP1[ob + y * POOL1_OUTPUT_SIZE + x] = m;
      }
    }
  }

  // Conv2
  for (int f = 0; f < CONV2_FILTERS; f++) {
    int ob = f * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    for (int y = 0; y < CONV2_OUTPUT_SIZE; y++) {
      for (int x = 0; x < CONV2_OUTPUT_SIZE; x++) {
        int32_t a = myL2.bias[f];
        for (int c = 0; c < CONV1_FILTERS; c++) {
          int ib = c * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
          for (int ky = 0; ky < 3; ky++) {
            const int8_t* ip = myP1 + ib + (y + ky) * POOL1_OUTPUT_SIZE + x;
            const int8_t* wp = myL2.w + f * 9 * CONV1_FILTERS + c * 9 + ky * 3;
            a += ip[0] * wp[0] + ip[1] * wp[1] + ip[2] * wp[2];
          }
        }
        int32_t r = (a >= 0) ? myRq(a, myL2.mp[f], myL2.sp[f]) : myRq(a, myL2.mn[f], myL2.sn[f]);
        myC2[ob + y * CONV2_OUTPUT_SIZE + x] = myClamp8(myL2.zp + r);
      }
    }
  }

  // Dense layer: int32 sum, requantize to an INT8 logit
  float mx = -1e30f;
  float real[NUM_CLASSES];
  for (int c = 0; c < NUM_CLASSES; c++) {
    int32_t a = myLD.bias[c];
    const int8_t* wp = myLD.w + (size_t)c * FLATTENED_SIZE;
    for (int i = 0; i < FLATTENED_SIZE; i++) a += (int32_t)myC2[i] * wp[i];
    myLogitsQ[c] = myClamp8(myLD.zp + myRq(a, myLD.mp[c], myLD.sp[c]));
    real[c] = ((int)myLogitsQ[c] - myLogitZp) * myLogitScale;
    if (real[c] > mx) mx = real[c];
  }

  // Softmax: turn the scores into probabilities that add up to 1
  float expSum = 0;
  for (int c = 0; c < NUM_CLASSES; c++) expSum += exp(real[c] - mx);
  for (int c = 0; c < NUM_CLASSES; c++) myProbs[c] = exp(real[c] - mx) / expSum;
}


// =============================================================================
//  MENU ITEM NUM_CLASSES+1 : INFER (LIVE CLASSIFICATION)
//  Every camera frame is shrunk to INPUT_SIZE and run through the INT8 network.
//  Serial prints the whole frame time and the network-only time in microseconds.
//  Exit with 't' or 'l', or touch the pad.
// =============================================================================
void myActionInfer() {
  if (!myModelLoaded) {
    Serial.println("ERROR: no INT8 model in memory.");
    Serial.println("Put myWeightsInt8.bin from the web page on the SD card and reboot,");
    Serial.println("or bake in myWeightsInt8.h. If the boot log said REFUSED, make the settings match and reflash.");
    myShowMessage("No model!", "Convert on the", "web page first");
    delay(3000);
    myResetMenuState();
    return;
  }
  if (!myCameraOk) {
    Serial.println("Camera not working - cannot infer");
    myShowMessage("Camera error");
    delay(2000);
    myResetMenuState();
    return;
  }

  Serial.println("\n>>> Inference mode (INT8)");
  Serial.println("  Exit: serial 't' or 'l', or touch the pad");

  myResetTouchState();

  int frameIndex = 0;                         // counts frames 0..9 for the OLED / debug update
  int pred = 0;
  unsigned long myInferCount = 0;             // frame number sent to the web page
  unsigned long inferStart = millis();

  while (true) {
    unsigned long frameStart = millis();

    if (Serial.available()) {
      char c = Serial.read();
      if (myHandleDebugChar(c)) {
        // handled: the web page is asking for debug frames
      } else if (c == 't' || c == 'T' || c == 'l' || c == 'L') {
        myResetMenuState();
        return;
      }
    }

    // A touch exits. Ignored for the first 1.5 s so the touch that opened this mode cannot close it.
    if (millis() - inferStart > 1500 && myReadTouch() > myThresholdPress) {
      Serial.println("Touch detected - exiting inference");
      delay(200);
      myResetMenuState();
      return;
    }

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera frame failed - retrying");
      delay(10);
      continue;
    }

    unsigned long netUs = 0;
    if (fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myRgbBuffer)) {
      myResizeToInput(myRgbBuffer, myIn);

      unsigned long netStart = micros();
      myForwardInt8();
      netUs = micros() - netStart;
      myInferCount++;

      pred = 0;
      for (int i = 1; i < NUM_CLASSES; i++) {
        if (myProbs[i] > myProbs[pred]) pred = i;
      }

      // Every 10th frame: update the OLED and the web page (myRgbBuffer is still valid here).
      if (frameIndex == 9) {
        myDebugSendFrame('I', (int)myInferCount, fb, pred, myProbs);
        int oW = u8g2.getDisplayWidth();
        int oH = u8g2.getDisplayHeight();
        int scX = 240 / oW;
        int scY = 240 / oH;
        u8g2.firstPage();
        do {
          for (int ox = 0; ox < oW; ox++) {
            for (int oy = 0; oy < oH; oy++) {
              int pi = ((oy * scY) * 240 + (ox * scX)) * 3;
              uint8_t bright = (myRgbBuffer[pi] + myRgbBuffer[pi + 1] + myRgbBuffer[pi + 2]) / 3;
              if (bright > 100) u8g2.drawPixel(ox, oy);
            }
          }
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setColorIndex(0);
          u8g2.drawBox(0, oH - 9, oW, 9);
          u8g2.setColorIndex(1);
          char buf[20];
          snprintf(buf, sizeof(buf), "%s %d%%", myClassLabels[pred].c_str(), (int)(myProbs[pred] * 100));
          u8g2.drawStr(1, oH - 1, buf);
        } while (u8g2.nextPage());
      }
    }

    esp_camera_fb_return(fb);

    unsigned long frameMs = millis() - frameStart;
    Serial.printf("Frame %d: %lu ms (%.1f FPS) network %lu us Current Pred: %s (%.1f%%) | All:",
                  frameIndex + 1, frameMs, 1000.0 / max(frameMs, 1UL), netUs,
                  myClassLabels[pred].c_str(), myProbs[pred] * 100);
    for (int i = 0; i < NUM_CLASSES; i++) Serial.printf(" %.0f%%", myProbs[i] * 100);
    Serial.println();

    frameIndex++;
    if (frameIndex >= 10) frameIndex = 0;
  }
}


// =============================================================================
//  MENU
//  Items 1..NUM_CLASSES collect images, then Infer.
//  Called from loop().
// =============================================================================

// Text for menu item 1..myTotalItems.
String myMenuLabel(int item) {
  if (item <= NUM_CLASSES) return myClassLabels[item - 1];
  return "Infer";
}

void myResetMenuState() {
  myIsSelected = false;
  myResetTouchState();
  myLastActivityTime = millis();
  myDrawMenu();
}

void myDrawMenu() {
  // ---- Serial menu ----
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    Serial.print(i == myMenuIndex ? " > " : "   ");
    Serial.printf("%d. %s\n", i, myMenuLabel(i).c_str());
  }
  Serial.println("Keys: t = next (tap)   l = select (long press)   or press the item's number");

  // ---- OLED menu: three rows, scrolled so the chosen item is always visible ----
  int myStartItem = myMenuIndex - 1;
  if (myStartItem > myTotalItems - 2) myStartItem = myTotalItems - 2;
  if (myStartItem < 1) myStartItem = 1;

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0, 7, "TAP:next 3x:OK");
    u8g2.setFont(u8g2_font_6x10_tf);
    for (int i = 0; i < 3; i++) {
      int cur = myStartItem + i;
      if (cur > myTotalItems) break;
      String line = (cur == myMenuIndex ? "> " : "  ") + myMenuLabel(cur);
      u8g2.drawStr(0, 18 + i * 9, line.c_str());
    }
  } while (u8g2.nextPage());
}

void myExecuteMenuItem(int idx) {
  if (idx <= NUM_CLASSES)        myActionCollect(idx - 1);
  else                           myActionInfer();
}

void myHandleMenuNavigation() {
  unsigned long myCurrentMillis = millis();

  // ---- Serial keys ----
  if (!myIsSelected && Serial.available()) {
    char c = Serial.read();

    if (myHandleDebugChar(c)) {
      // handled: the web page is asking for debug frames
    }
    else if (c >= '1' && c <= '9') {            // jump straight to an item
      int newIndex = c - '0';
      if (newIndex <= myTotalItems) {
        myMenuIndex = newIndex;
        myIsSelected = true;
        myLastActivityTime = myCurrentMillis;
        myExecuteMenuItem(myMenuIndex);
      }
    }
    else if (c == 't' || c == 'T') {            // next item
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (c == 'l' || c == 'L') {            // select
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }

  // ---- Touch pad ----
  if (!myIsSelected) {
    int touchAction = myCheckTouchInput();

    if (touchAction == 1) {                     // tap: next item
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (touchAction == 2) {                // 3+ taps: select
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }
}
