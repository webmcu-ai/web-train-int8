// =============================================================================
//  XIAO ESP32S3 SENSE / XIAO ML KIT  -  VISION: TRAIN FLOAT, CONVERT TO INT8, INFER  -  firmware-v006
//
//  Collect small images, train a tiny CNN in float32 on the device, convert it to INT8
//  on the device, and classify what the camera sees with either model. The web page can
//  do the same steps faster; models move between the page and the device on the SD card.
//
//  By Jeremy Ellis   (MIT license, use at your own risk)
//  With free tier assistance from Claude (code overview), ChatGPT (critique),
//  Gemini (research) and Copilot (alternate).
//  GitHub   https://github.com/hpssjellis
//  LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// -----------------------------------------------------------------------------
//  THE MENU  (Serial monitor at 115200 baud and the 72x40 OLED)
//    1..NUM_CLASSES    collect images for that class (one folder per class)
//    NUM_CLASSES+1     Train FLOAT        (trains the float model on the device)
//    NUM_CLASSES+2     Convert to INT8    (float model -> INT8 model, saved to the SD card)
//    NUM_CLASSES+3     Infer INT8         (live camera classification, INT8 model)
//    NUM_CLASSES+4     Infer FLOAT        (live camera classification, float model)
//    NUM_CLASSES+5     Bench MobileNet    (times MobileNetV1 alpha 0.25 layer shapes with synthetic INT8 weights at 128 and 96 input:
//                                          scalar vs fast loops, PSRAM vs internal RAM, one vs two cores, with matching output checksums;
//                                          prints time and memory only, it does not classify and needs no SD card;
//                                          the kernels are compiled with -O3 when MY_BENCH_O3 is 1)
//
//  CONTROLS
//    Touch pad on pin A0:   1-2 taps = next item / capture / train again
//                           3+ taps  = select / exit back to the menu
//    Serial monitor keys:   t = next / capture / train again      l = select / exit
//                           1..9 = jump straight to a menu item (menu only)
//                           s = switch between INT8 and float while inferring
//                           x or l = stop training or converting
//
// -----------------------------------------------------------------------------
//  THE SD CARD
//    /images/<class name>/img_*.jpg    240x240 photos, one folder per class
//    /header/myWeights.bin             float model (written by Train FLOAT or by the web page)
//    /header/myWeightsInt8.bin         INT8 model (written by Convert to INT8 or by the web page)
//    /header/myWeightsInt8.h           same INT8 bytes as a C array (for baking into the sketch)
//    /header/config.json               class names and "grayscale" true/false, written by the web page
//
// -----------------------------------------------------------------------------
//  WORKFLOWS  (any mix works, both sides read and write the same files)
//   A) All on the device:  collect images -> Train FLOAT -> Convert to INT8 -> Infer.
//   B) Train on the web page: collect images, move the SD card to the computer, train and
//      convert on the page, press "Save both models", put the card back, reboot, Infer.
//   C) Train on the page, convert on the device: save only the float model from the page,
//      choose Convert to INT8 here.
//  A bigger model than the device can train (for example 96x96 input) still runs here:
//  Train FLOAT then says "Not enough memory" but Convert and Infer can still work.
//
//  Training changes the float weights, so the old INT8 model no longer belongs to them.
//  After Train FLOAT the old INT8 files are deleted; choose Convert to INT8 again.
//
// -----------------------------------------------------------------------------
//  HOW THE CONVERSION WORKS
//    1. The float model classifies up to CALIBRATION_IMAGES training images and the smallest
//       and largest value seen in conv1, conv2 and the output scores are recorded.
//    2. Each layer's weights become int8 (one scale per filter) and each layer's activations
//       become int8 using the recorded range. Biases and sums are int32.
//    3. One image and its expected INT8 output are stored in the file for the boot self-test.
//    4. Float and INT8 are compared on the validation images and the result is printed.
//  The math is the same as the web page, so the same float weights and ranges give the same file.
//
// -----------------------------------------------------------------------------
//  SETTINGS THAT MUST MATCH THE WEB PAGE  (fixed when the sketch compiles)
//    INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES and myClassLabels[]
//  The page shows the exact lines to paste (section 2 "Classes and data").
//  Grayscale is NOT compiled in: it comes from "grayscale" in config.json.
//  If anything differs the device prints "REFUSED" with both sets of numbers.
//
//  INT8 WEIGHTS PRIORITY:  SD card myWeightsInt8.bin  >  baked-in myWeightsInt8.h
//  To bake the INT8 weights in: copy /header/myWeightsInt8.h next to this sketch, set the
//  settings above to match, then remove the // in front of  #define USE_BAKED_WEIGHTS
//  The float model is always read from the SD card.
//
//  BOOT CHECKS
//    The INT8 file holds one test image and its expected INT8 output. At boot the
//    device runs it and prints PASS or FAIL. When both models are loaded, the float
//    model also classifies that same test image and the two answers are compared.
//
// -----------------------------------------------------------------------------
//  WEB PAGE DEBUG FRAMES
//  Nothing extra is printed unless the web page sends 'D' (it repeats it every
//  5 s; the device stops after 15 s of silence, or when it receives 'd').
//  While on, the device prints "@F ..." lines holding the camera JPEG (base64),
//  the class probabilities, a heatmap and which model made them (I8 or F32), so
//  the page can compare the same model on exactly the same picture.
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
#include "esp_heap_caps.h"
#include "freertos/semphr.h"


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

// ---- Training settings (Train FLOAT) ------------------------------------------
float LEARNING_RATE     = 0.0003;
int   BATCH_SIZE        = 6;
int   TARGET_EPOCHS     = 20;
int   VALIDATION_IMAGES = 3;     // last N images per class are held out to measure accuracy (0 = off)

// ---- Conversion settings (Convert to INT8) ------------------------------------
int   CALIBRATION_IMAGES = 100;  // how many training images are used to measure the value ranges

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

// Weight counts (conv1 depends on the channel count read from config.json)
#define MY_CONV1_WEIGHTS  (CONV1_FILTERS * 9 * myChannels)
#define MY_CONV2_WEIGHTS  (CONV2_FILTERS * 9 * CONV1_FILTERS)
#define MY_DENSE_WEIGHTS  (FLATTENED_SIZE * NUM_CLASSES)

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
bool myFloatLoaded = false;       // true once a float model is in memory
int  myActiveEngine = 0;          // model used by the current inference: 0 = INT8, 1 = float

// ---- Input channels (from config.json, 3 = RGB, 1 = grayscale) ----------------
bool myConfigHasGray = false;
int  myChannels = 3;

// ---- Menu ---------------------------------------------------------------------
const int myTotalItems = NUM_CLASSES + 5;   // one per class, then Train FLOAT, Convert to INT8, Infer INT8, Infer FLOAT, Bench MobileNet

// ---- Bench MobileNet: MobileNetV1 alpha 0.25 layer shapes (13 depthwise + pointwise blocks) ----
const int myBenchSizes[2] = {128, 96};        // input sizes that are timed (each divisible by 32)
const int myBenchPasses = 2;                  // timed passes per variant
#define MY_BENCH_O3 1                         // 1 = bench kernels compiled with -O3, 0 = default build optimization
const int myBenchO3 = MY_BENCH_O3;
struct MyBenchBlock { uint16_t outCh; uint8_t stride; };
const MyBenchBlock myBenchBlocks[13] = { {16,1}, {32,2}, {32,1}, {64,2}, {64,1}, {128,2}, {128,1}, {128,1}, {128,1}, {128,1}, {128,1}, {256,2}, {256,1} };
const char* myBenchConfigNames[3] = {"PSRAM acts + PSRAM weights", "internal acts + PSRAM weights", "internal acts + internal weights"};
const char* myBenchModeNames[3] = {"scalar", "fast", "fast + 2 cores"};
struct MyBenchJob { int kind; const int8_t* in; int8_t* out; const int8_t* w; int size; int ch; int stride; int inCh; int outCh; int from; int to; };
MyBenchJob myBenchJob;
SemaphoreHandle_t myBenchStartSem = NULL;
SemaphoreHandle_t myBenchDoneSem = NULL;
TaskHandle_t myBenchWorker = NULL;
volatile bool myBenchWorkerQuit = false;
const int8_t* myBenchOutPtr = NULL;
size_t myBenchOutLen = 0;
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

// ---- Float model (PSRAM, allocated only when myWeights.bin is found) ----------
float* myFW1 = nullptr;                    // conv1 weights
float* myFB1 = nullptr;
float* myFW2 = nullptr;                    // conv2 weights
float* myFB2 = nullptr;
float* myFWD = nullptr;                    // dense weights
float* myFBD = nullptr;
float* myInF = nullptr;                    // model input, pixel / 255
float* myFC1 = nullptr;
float* myFP1 = nullptr;
float* myFC2 = nullptr;
float  myFLogits[NUM_CLASSES];

// ---- INT8 buffers -------------------------------------------------------------
int8_t* myIn = nullptr;                    // model input, pixel - 128
int8_t* myC1 = nullptr;
int8_t* myP1 = nullptr;
int8_t* myC2 = nullptr;
int8_t  myLogitsQ[NUM_CLASSES];
float   myProbs[NUM_CLASSES];

// ---- Training buffers (PSRAM, allocated only when Train FLOAT is opened) -------
bool myTrainingBuffersReady = false;
float* myGW1 = nullptr;                    // gradients
float* myGB1 = nullptr;
float* myGW2 = nullptr;
float* myGB2 = nullptr;
float* myGWD = nullptr;
float* myGBD = nullptr;
float* myMW1 = nullptr;                    // Adam memory m
float* myMB1 = nullptr;
float* myMW2 = nullptr;
float* myMB2 = nullptr;
float* myMWD = nullptr;
float* myMBD = nullptr;
float* myVW1 = nullptr;                    // Adam memory v
float* myVB1 = nullptr;
float* myVW2 = nullptr;
float* myVB2 = nullptr;
float* myVWD = nullptr;
float* myVBD = nullptr;
float* myDenseGrad = nullptr;              // back-propagation work buffers
float* myConv2Grad = nullptr;
float* myPool1Grad = nullptr;
float* myConv1Grad = nullptr;

// One entry per training buffer, used by myAllocateTrainingBuffers().
struct MyBufSpec {
  float** ptr;
  size_t count;
  bool zero;
};

// One image on the SD card and its class number.
struct MyImageItem {
  String path;
  int label;
};

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
void   myCrossCheck();

// float model
size_t myExpectedFloatBytes();
void   myFreeFloatMemory();
bool   myAllocateFloatMemory();
bool   myLoadFloatWeights();
void   myInitFloatRandom();
void   mySaveFloatWeights();
void   myInvalidateInt8();

// float training
inline float myClipValue(float v, float mn, float mx);
inline float myLeakyDeriv(float x);
size_t myTrainingBytesNeeded();
bool   myAllocateTrainingBuffers();
void   myResetAdam();
bool   myLoadImageFile(const char* path);
void   myBuildImageList(std::vector<MyImageItem>& all, std::vector<MyImageItem>& train, std::vector<MyImageItem>& val);
void   myBackwardDense(int label);
void   myBackwardConv2();
void   myBackwardPool1();
void   myBackwardConv1();
void   myAdamUpdate(float* w, float* g, float* m, float* v, int size, int step);
void   myUpdateWeights(int step);
bool   myTrainingExitRequested();
void   myActionTrain();

// INT8 conversion
void   myQuantMult(double m, int32_t& mult, int32_t& shift);
void   myActParams(double lo, double hi, double& scale, int32_t& zp);
void   myQuantLayer(const float* w, const float* b, int count, int per, double sx, int zpx, double os, int ozp, bool leaky, MyQLayer& L, double& worst);
bool   mySaveInt8Files();
void   myActionConvert();

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
void myForwardFloat();
void myFillFloatInput();
float myHeatVal(int i);
int  myArgmaxProbs();

// menu actions
void myActionCollect(int classIdx);
void myActionInfer(int engine);
void myActionBenchMobileNet();
void myBenchFirstConvRows(const int8_t* in, int8_t* out, const int8_t* w, int size, int y0, int y1);
void myBenchFirstConv(const int8_t* in, int8_t* out, const int8_t* w, int size);
void myBenchDepthwise(const int8_t* in, int8_t* out, const int8_t* w, int size, int ch, int stride);
void myBenchPointwise(const int8_t* in, int8_t* out, const int8_t* w, int pixels, int inCh, int outCh);
void myBenchDepthwiseFast(const int8_t* in, int8_t* out, const int8_t* w, int size, int ch, int stride, int y0, int y1);
void myBenchPointwiseFast(const int8_t* in, int8_t* out, const int8_t* w, int inCh, int outCh, int p0, int p1);
int8_t myBenchClamp(int32_t acc);
void myBenchDoJob(const MyBenchJob* j);
void myBenchWorkerTask(void* arg);
void myBenchLayer(MyBenchJob job, int total, bool dual);
double myBenchRun(int8_t* myA, int8_t* myB, const int8_t* myW, int res, int mode);
uint32_t myBenchSum(const int8_t* p, size_t n);
size_t myBenchWeightTotal();

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

  Serial.println("\n=== XIAO ESP32S3 Vision, train, convert and infer (firmware-v006) ===");
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

  if (myLoadFloatWeights()) {
    Serial.println("SD float weights loaded");
  }

  Serial.printf("Input channels: %d (%s)\n", myChannels, myChannels == 1 ? "grayscale" : "RGB");
  mySelfTest();
  myCrossCheck();
  Serial.printf("Models ready: INT8 %s, float %s\n", myModelLoaded ? "yes" : "no", myFloatLoaded ? "yes" : "no");

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


// =============================================================================
//  THE FLOAT MODEL FILE  (/header/myWeights.bin)
//  Plain little endian floats in this order: conv1 weights, conv1 bias, conv2 weights,
//  conv2 bias, dense weights, dense bias.
// =============================================================================

// Exact size in bytes of myWeights.bin for this layout and channel count.
size_t myExpectedFloatBytes() {
  size_t n = (size_t)CONV1_FILTERS * 9 * myChannels + CONV1_FILTERS;
  n += (size_t)CONV2_FILTERS * 9 * CONV1_FILTERS + CONV2_FILTERS;
  n += (size_t)FLATTENED_SIZE * NUM_CLASSES + NUM_CLASSES;
  return n * 4;
}

void myFreeFloatMemory() {
  float** all[10] = { &myFW1, &myFB1, &myFW2, &myFB2, &myFWD, &myFBD, &myInF, &myFC1, &myFP1, &myFC2 };
  for (int i = 0; i < 10; i++) {
    if (*all[i]) { free(*all[i]); *all[i] = nullptr; }
  }
}

bool myAllocateFloatMemory() {
  if (myFW1) return true;
  myFW1 = (float*)ps_malloc((size_t)CONV1_FILTERS * 9 * myChannels * 4);
  myFB1 = (float*)ps_malloc(CONV1_FILTERS * 4);
  myFW2 = (float*)ps_malloc((size_t)CONV2_FILTERS * 9 * CONV1_FILTERS * 4);
  myFB2 = (float*)ps_malloc(CONV2_FILTERS * 4);
  myFWD = (float*)ps_malloc((size_t)FLATTENED_SIZE * NUM_CLASSES * 4);
  myFBD = (float*)ps_malloc(NUM_CLASSES * 4);
  myInF = (float*)ps_malloc((size_t)INPUT_SIZE * INPUT_SIZE * myChannels * 4);
  myFC1 = (float*)ps_malloc((size_t)CONV1_OUT_BYTES * 4);
  myFP1 = (float*)ps_malloc((size_t)POOL1_OUT_BYTES * 4);
  myFC2 = (float*)ps_malloc((size_t)FLATTENED_SIZE * 4);
  bool ok = myFW1 && myFB1 && myFW2 && myFB2 && myFWD && myFBD && myInF && myFC1 && myFP1 && myFC2;
  if (!ok) myFreeFloatMemory();
  return ok;
}

// Loads /header/myWeights.bin from the SD card. Refuses a file of the wrong size.
bool myLoadFloatWeights() {
  if (!mySDavailable) return false;
  if (!SD.exists("/header/myWeights.bin")) {
    Serial.println("No SD float weights file found");
    return false;
  }
  File f = SD.open("/header/myWeights.bin", FILE_READ);
  if (!f) return false;
  size_t need = myExpectedFloatBytes();
  if ((size_t)f.size() != need) {
    Serial.printf("REFUSED float weights: file is %u bytes but this sketch needs %u bytes\n", (unsigned)f.size(), (unsigned)need);
    Serial.printf("Sketch layout: INPUT_SIZE %d, CONV1_FILTERS %d, CONV2_FILTERS %d, NUM_CLASSES %d, channels %d\n",
                  INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES, myChannels);
    Serial.println("Make the settings match the web page, reflash, or save the model again.");
    f.close();
    return false;
  }
  if (!myAllocateFloatMemory()) {
    Serial.println("Not enough PSRAM for the float model - INT8 still works. Use a smaller INPUT_SIZE.");
    f.close();
    return false;
  }
  bool ok = f.read((uint8_t*)myFW1, (size_t)CONV1_FILTERS * 9 * myChannels * 4) > 0;
  ok = ok && f.read((uint8_t*)myFB1, CONV1_FILTERS * 4) > 0;
  ok = ok && f.read((uint8_t*)myFW2, (size_t)CONV2_FILTERS * 9 * CONV1_FILTERS * 4) > 0;
  ok = ok && f.read((uint8_t*)myFB2, CONV2_FILTERS * 4) > 0;
  ok = ok && f.read((uint8_t*)myFWD, (size_t)FLATTENED_SIZE * NUM_CLASSES * 4) > 0;
  ok = ok && f.read((uint8_t*)myFBD, NUM_CLASSES * 4) > 0;
  f.close();
  if (!ok) {
    Serial.println("Float weights file could not be read");
    myFreeFloatMemory();
    return false;
  }
  myFloatLoaded = true;
  Serial.printf("Free PSRAM after the float model: %d bytes\n", ESP.getFreePsram());
  return true;
}

// With both models loaded, the float model classifies the INT8 file's test image and the answers are compared.
void myCrossCheck() {
  if (!myHasTest || !myModelLoaded || !myFloatLoaded) return;
  int total = INPUT_SIZE * INPUT_SIZE * myChannels;
  for (int i = 0; i < total; i++) myIn[i] = (int8_t)((int)myTestImage[i] - 128);
  myForwardInt8();
  int predInt8 = myArgmaxProbs();
  float probInt8 = myProbs[predInt8];
  myFillFloatInput();
  myForwardFloat();
  int predFloat = myArgmaxProbs();
  Serial.printf("Test image: float says %s %.1f%%, INT8 says %s %.1f%% -> %s\n",
                myClassLabels[predFloat].c_str(), myProbs[predFloat] * 100,
                myClassLabels[predInt8].c_str(), probInt8 * 100,
                predFloat == predInt8 ? "the models agree" : "the models DISAGREE");
}


// =============================================================================
//  FLOAT TRAINING  (menu item NUM_CLASSES+1)
//  Trains the float model on the images in /images/<class>/ with Adam, then writes
//  /header/myWeights.bin. The network and its file order are the same as the web page.
// =============================================================================

inline float myClipValue(float v, float mn, float mx) {
  return v < mn ? mn : (v > mx ? mx : v);
}
inline float myLeakyDeriv(float x) { return x > 0 ? 1.0f : 0.1f; }

// Random starting weights (He initialization) for a model that has no file yet.
void myInitFloatRandom() {
  float c1std = sqrt(2.0 / (9.0 * myChannels));
  for (int i = 0; i < MY_CONV1_WEIGHTS; i++) myFW1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c1std;
  for (int i = 0; i < CONV1_FILTERS; i++) myFB1[i] = 0;
  float c2std = sqrt(2.0 / (9.0 * CONV1_FILTERS));
  for (int i = 0; i < MY_CONV2_WEIGHTS; i++) myFW2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c2std;
  for (int i = 0; i < CONV2_FILTERS; i++) myFB2[i] = 0;
  float dstd = sqrt(2.0 / FLATTENED_SIZE);
  for (int i = 0; i < MY_DENSE_WEIGHTS; i++) myFWD[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * dstd;
  for (int i = 0; i < NUM_CLASSES; i++) myFBD[i] = 0;
}

// Writes /header/myWeights.bin (conv1 w, conv1 b, conv2 w, conv2 b, dense w, dense b).
void mySaveFloatWeights() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot save weights");
    return;
  }
  if (!SD.exists("/header")) SD.mkdir("/header");
  if (SD.exists("/header/myWeights.bin")) SD.remove("/header/myWeights.bin");
  File f = SD.open("/header/myWeights.bin", FILE_WRITE);
  if (!f) {
    Serial.println("Could not open /header/myWeights.bin for writing");
    return;
  }
  f.write((uint8_t*)myFW1, (size_t)MY_CONV1_WEIGHTS * 4);
  f.write((uint8_t*)myFB1, CONV1_FILTERS * 4);
  f.write((uint8_t*)myFW2, (size_t)MY_CONV2_WEIGHTS * 4);
  f.write((uint8_t*)myFB2, CONV2_FILTERS * 4);
  f.write((uint8_t*)myFWD, (size_t)MY_DENSE_WEIGHTS * 4);
  f.write((uint8_t*)myFBD, NUM_CLASSES * 4);
  f.close();
  Serial.println("Float weights saved to /header/myWeights.bin");
}

// The INT8 model was made from the old float weights, so it is dropped from memory and from the SD card.
void myInvalidateInt8() {
  myModelLoaded = false;
  myHasTest = false;
  if (!mySDavailable) return;
  bool removed = false;
  if (SD.exists("/header/myWeightsInt8.bin")) { SD.remove("/header/myWeightsInt8.bin"); removed = true; }
  if (SD.exists("/header/myWeightsInt8.h"))   { SD.remove("/header/myWeightsInt8.h");   removed = true; }
  if (removed) Serial.println("The old INT8 files were deleted because the float weights changed. Choose Convert to INT8.");
}

// Bytes of PSRAM that the training-only buffers need.
size_t myTrainingBytesNeeded() {
  size_t floats = (size_t)(MY_CONV1_WEIGHTS + CONV1_FILTERS + MY_CONV2_WEIGHTS + CONV2_FILTERS + MY_DENSE_WEIGHTS + NUM_CLASSES) * 3;  // gradient + Adam m + Adam v
  floats += (size_t)FLATTENED_SIZE * 2 + POOL1_OUT_BYTES + CONV1_OUT_BYTES;                                                            // back-propagation buffers
  return floats * sizeof(float);
}

// Allocates the training buffers once. Returns false (and frees anything partly allocated)
// when there is not enough PSRAM. Inference and conversion are never affected.
bool myAllocateTrainingBuffers() {
  if (myTrainingBuffersReady) return true;

  size_t need = myTrainingBytesNeeded();
  if (ESP.getFreePsram() < need + 32768) {
    Serial.printf("Training needs about %u KB of PSRAM but only %u KB is free.\n",
                  (unsigned)(need / 1024), (unsigned)(ESP.getFreePsram() / 1024));
    return false;
  }

  MyBufSpec specs[] = {
    { &myGW1, (size_t)MY_CONV1_WEIGHTS, false }, { &myGB1, CONV1_FILTERS, false },
    { &myGW2, (size_t)MY_CONV2_WEIGHTS, false }, { &myGB2, CONV2_FILTERS, false },
    { &myGWD, (size_t)MY_DENSE_WEIGHTS, false }, { &myGBD, NUM_CLASSES,   false },
    { &myMW1, (size_t)MY_CONV1_WEIGHTS, true },  { &myVW1, (size_t)MY_CONV1_WEIGHTS, true },
    { &myMB1, CONV1_FILTERS, true },             { &myVB1, CONV1_FILTERS, true },
    { &myMW2, (size_t)MY_CONV2_WEIGHTS, true },  { &myVW2, (size_t)MY_CONV2_WEIGHTS, true },
    { &myMB2, CONV2_FILTERS, true },             { &myVB2, CONV2_FILTERS, true },
    { &myMWD, (size_t)MY_DENSE_WEIGHTS, true },  { &myVWD, (size_t)MY_DENSE_WEIGHTS, true },
    { &myMBD, NUM_CLASSES, true },               { &myVBD, NUM_CLASSES, true },
    { &myDenseGrad, FLATTENED_SIZE, false },
    { &myConv2Grad, FLATTENED_SIZE, false },
    { &myPool1Grad, POOL1_OUT_BYTES, false },
    { &myConv1Grad, CONV1_OUT_BYTES, false }
  };
  const int n = sizeof(specs) / sizeof(specs[0]);

  bool ok = true;
  for (int i = 0; i < n; i++) {
    *specs[i].ptr = specs[i].zero ? (float*)ps_calloc(specs[i].count, sizeof(float))
                                  : (float*)ps_malloc(specs[i].count * sizeof(float));
    if (!*specs[i].ptr) { ok = false; break; }
  }
  if (!ok) {
    Serial.println("Training buffer allocation failed - freeing what was allocated");
    for (int i = 0; i < n; i++) {
      if (*specs[i].ptr) { free(*specs[i].ptr); *specs[i].ptr = nullptr; }
    }
    return false;
  }

  myTrainingBuffersReady = true;
  Serial.printf("Training buffers allocated. Free PSRAM: %u KB\n", (unsigned)(ESP.getFreePsram() / 1024));
  return true;
}

// Forgets the optimizer memory (done at the start of every training run).
void myResetAdam() {
  memset(myMW1, 0, (size_t)MY_CONV1_WEIGHTS * 4); memset(myVW1, 0, (size_t)MY_CONV1_WEIGHTS * 4);
  memset(myMB1, 0, CONV1_FILTERS * 4);            memset(myVB1, 0, CONV1_FILTERS * 4);
  memset(myMW2, 0, (size_t)MY_CONV2_WEIGHTS * 4); memset(myVW2, 0, (size_t)MY_CONV2_WEIGHTS * 4);
  memset(myMB2, 0, CONV2_FILTERS * 4);            memset(myVB2, 0, CONV2_FILTERS * 4);
  memset(myMWD, 0, (size_t)MY_DENSE_WEIGHTS * 4); memset(myVWD, 0, (size_t)MY_DENSE_WEIGHTS * 4);
  memset(myMBD, 0, NUM_CLASSES * 4);              memset(myVBD, 0, NUM_CLASSES * 4);
}

// Reads one JPEG from the SD card and fills both model inputs (INT8 myIn and float myInF).
bool myLoadImageFile(const char* path) {
  File f = SD.open(path);
  if (!f) return false;

  size_t sz = f.size();
  uint8_t* jpg = (uint8_t*)ps_malloc(sz);
  if (!jpg) { f.close(); return false; }
  f.read(jpg, sz);
  f.close();

  bool ok = fmt2rgb888(jpg, sz, PIXFORMAT_JPEG, myRgbBuffer);
  free(jpg);
  if (!ok) return false;

  myResizeToInput(myRgbBuffer, myIn);
  myFillFloatInput();
  return true;
}

// Lists every image, sorted by path, and splits off the last VALIDATION_IMAGES of each class.
void myBuildImageList(std::vector<MyImageItem>& all, std::vector<MyImageItem>& train, std::vector<MyImageItem>& val) {
  all.clear(); train.clear(); val.clear();
  for (int i = 0; i < NUM_CLASSES; i++) {
    File root = SD.open("/images/" + myClassLabels[i]);
    if (root) {
      while (File file = root.openNextFile()) {
        if (!file.isDirectory()) {
          String fn = String(file.name());
          if (fn.endsWith(".jpg") || fn.endsWith(".JPG")) all.push_back({file.path(), i});
        }
        file.close();
      }
      root.close();
    }
  }
  std::sort(all.begin(), all.end(), [](const MyImageItem& a, const MyImageItem& b) { return a.path < b.path; });

  int counts[NUM_CLASSES] = {};
  for (auto& item : all) counts[item.label]++;
  int skip[NUM_CLASSES];
  for (int c = 0; c < NUM_CLASSES; c++) skip[c] = VALIDATION_IMAGES > 0 ? min(VALIDATION_IMAGES, counts[c]) : 0;

  int seen[NUM_CLASSES] = {};
  for (int i = (int)all.size() - 1; i >= 0; i--) {
    int c = all[i].label;
    if (seen[c] < skip[c]) { val.push_back(all[i]); seen[c]++; }
    else                   { train.push_back(all[i]); }
  }
  std::reverse(train.begin(), train.end());
  std::reverse(val.begin(), val.end());
}

// ---- Backward pass: how much should each weight change? -----------------------
// Weight gradients are ADDED over the images of a batch and cleared once per batch.

void myBackwardDense(int label) {
  memset(myDenseGrad, 0, FLATTENED_SIZE * sizeof(float));
  for (int c = 0; c < NUM_CLASSES; c++) {
    float error = myProbs[c] - (c == label ? 1.0f : 0.0f);
    for (int i = 0; i < FLATTENED_SIZE; i++) {
      myGWD[c * FLATTENED_SIZE + i] += error * myFC2[i];
      myDenseGrad[i] += error * myFWD[c * FLATTENED_SIZE + i];
    }
    myGBD[c] += error;
  }
}

void myBackwardConv2() {
  for (int i = 0; i < FLATTENED_SIZE; i++) {
    myConv2Grad[i] = myDenseGrad[i] * myLeakyDeriv(myFC2[i]);
  }
  memset(myPool1Grad, 0, POOL1_OUT_BYTES * sizeof(float));

  for (int f = 0; f < CONV2_FILTERS; f++) {
    int ob = f * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    for (int y = 0; y < CONV2_OUTPUT_SIZE; y++) {
      for (int x = 0; x < CONV2_OUTPUT_SIZE; x++) {
        float grad = myConv2Grad[ob + y * CONV2_OUTPUT_SIZE + x];
        myGB2[f] += grad;
        for (int c = 0; c < CONV1_FILTERS; c++) {
          int ib = c * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
          for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
              int pi = ib + (y + ky) * POOL1_OUTPUT_SIZE + (x + kx);
              int wi = f * 9 * CONV1_FILTERS + c * 9 + ky * 3 + kx;
              myGW2[wi] += grad * myFP1[pi];
              myPool1Grad[pi] += grad * myFW2[wi];
            }
          }
        }
      }
    }
  }
}

// The gradient only flows back to the pixel that "won" each 2x2 max pool.
void myBackwardPool1() {
  memset(myConv1Grad, 0, CONV1_OUT_BYTES * sizeof(float));
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ib = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE, ob = f * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
    for (int y = 0; y < POOL1_OUTPUT_SIZE; y++) {
      for (int x = 0; x < POOL1_OUTPUT_SIZE; x++) {
        int iy = y * 2, ix = x * 2;
        float poolVal = myFP1[ob + y * POOL1_OUTPUT_SIZE + x];
        float grad = myPool1Grad[ob + y * POOL1_OUTPUT_SIZE + x];
        if (myFC1[ib + iy * CONV1_OUTPUT_SIZE + ix] == poolVal)             myConv1Grad[ib + iy * CONV1_OUTPUT_SIZE + ix] += grad;
        if (myFC1[ib + iy * CONV1_OUTPUT_SIZE + ix + 1] == poolVal)         myConv1Grad[ib + iy * CONV1_OUTPUT_SIZE + ix + 1] += grad;
        if (myFC1[ib + (iy + 1) * CONV1_OUTPUT_SIZE + ix] == poolVal)       myConv1Grad[ib + (iy + 1) * CONV1_OUTPUT_SIZE + ix] += grad;
        if (myFC1[ib + (iy + 1) * CONV1_OUTPUT_SIZE + ix + 1] == poolVal)   myConv1Grad[ib + (iy + 1) * CONV1_OUTPUT_SIZE + ix + 1] += grad;
      }
    }
  }
}

void myBackwardConv1() {
  const int ch = myChannels;
  for (int i = 0; i < CONV1_OUT_BYTES; i++) {
    myConv1Grad[i] *= myLeakyDeriv(myFC1[i]);
  }

  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    for (int y = 0; y < CONV1_OUTPUT_SIZE; y++) {
      for (int x = 0; x < CONV1_OUTPUT_SIZE; x++) {
        float grad = myConv1Grad[ob + y * CONV1_OUTPUT_SIZE + x];
        myGB1[f] += grad;
        for (int ky = 0; ky < 3; ky++) {
          for (int kx = 0; kx < 3; kx++) {
            const float* ip = myInF + ((y + ky) * INPUT_SIZE + (x + kx)) * ch;
            float* gp = myGW1 + f * 9 * ch + (ky * 3 + kx) * ch;
            for (int c = 0; c < ch; c++) gp[c] += grad * ip[c];
          }
        }
      }
    }
  }
}

// ---- Optimizer: Adam ----------------------------------------------------------
void myAdamUpdate(float* w, float* g, float* m, float* v, int size, int step) {
  float b1 = 0.9f, b2 = 0.999f, eps = 1e-6f;     // eps 1e-6 (not 1e-8) avoids NaN in 32-bit floats
  float lr_t = LEARNING_RATE * sqrt(1 - pow(b2, step)) / (1 - pow(b1, step));
  for (int i = 0; i < size; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    w[i] -= lr_t * m[i] / (sqrt(v[i]) + eps);
    w[i] = myClipValue(w[i], -10, 10);
  }
}

void myUpdateWeights(int step) {
  myAdamUpdate(myFW1, myGW1, myMW1, myVW1, MY_CONV1_WEIGHTS, step);
  myAdamUpdate(myFB1, myGB1, myMB1, myVB1, CONV1_FILTERS,    step);
  myAdamUpdate(myFW2, myGW2, myMW2, myVW2, MY_CONV2_WEIGHTS, step);
  myAdamUpdate(myFB2, myGB2, myMB2, myVB2, CONV2_FILTERS,    step);
  myAdamUpdate(myFWD, myGWD, myMWD, myVWD, MY_DENSE_WEIGHTS, step);
  myAdamUpdate(myFBD, myGBD, myMBD, myVBD, NUM_CLASSES,      step);
}

// True when the user asked to stop (serial 'x' or 'l', or 3+ taps on the touch pad).
bool myTrainingExitRequested() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
      Serial.println("Stopping...");
      return true;
    }
  }
  myCheckTouchBackground();
  if (myPeekTouchAction() == 2) {
    myCheckTouchInput();                // consume the long press
    Serial.println("Long press - stopping");
    return true;
  }
  return false;
}

void myActionTrain() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot train");
    myShowMessage("No SD card");
    delay(2000);
    myResetMenuState();
    return;
  }

  if (!myAllocateFloatMemory() || !myAllocateTrainingBuffers()) {
    Serial.println("Not enough memory to train this model size on the device.");
    Serial.println("Train on the web page instead (or lower INPUT_SIZE / filters). Convert and Infer can still work.");
    myShowMessage("Not enough", "memory to train", "Infer still OK");
    delay(3000);
    myResetMenuState();
    return;
  }

  Serial.println("\n>>> Train FLOAT");
  Serial.println("  During training: 3+ taps (or 'x') = save and exit");
  Serial.println("  When finished:   TAP (or 't') = train again, 3+ taps (or 'l') = exit");

  myResetTouchState();
  myShowMessage("TRAIN FLOAT", "Loading...");

  if (myFloatLoaded) {
    Serial.println("Continuing from the float weights already in memory");
  } else {
    myInitFloatRandom();
    Serial.println("Starting fresh with random weights");
  }

  while (true) {
    std::vector<MyImageItem> myAll, myTrain, myVal;
    myBuildImageList(myAll, myTrain, myVal);

    if (myTrain.empty()) {
      Serial.println("No images found in /images/<class>/");
      myShowMessage("No Images!");
      delay(2000);
      myResetMenuState();
      return;
    }
    Serial.printf("Validation: %d images  Train: %d images\n", (int)myVal.size(), (int)myTrain.size());

    int total = myTrain.size();
    int batchesPerEpoch = (total + BATCH_SIZE - 1) / BATCH_SIZE;
    int totalBatches = TARGET_EPOCHS * batchesPerEpoch;
    Serial.printf("Training: %d images, %d batches\n", total, totalBatches);

    myResetAdam();                       // the optimizer's step counter restarts, so its memory does too

    std::vector<int> indices;
    for (int i = 0; i < total; i++) indices.push_back(i);

    float runningLoss = 0;
    int lossCount = 0;
    bool myStopped = false;

    for (int batch = 0; batch < totalBatches && !myStopped; batch++) {
      if (myTrainingExitRequested()) { myStopped = true; break; }

      // Shuffle the image order at the start of each epoch.
      if (batch % batchesPerEpoch == 0) {
        int epoch = batch / batchesPerEpoch + 1;
        Serial.printf("\n--- Epoch %d/%d ---\n", epoch, TARGET_EPOCHS);
        for (int i = total - 1; i > 0; i--) {
          int j = random(i + 1);
          int tmp = indices[i];
          indices[i] = indices[j];
          indices[j] = tmp;
        }
      }

      int batchStart = (batch % batchesPerEpoch) * BATCH_SIZE;
      int batchEnd = min(batchStart + BATCH_SIZE, total);

      float batchLoss = 0;
      int correctCount = 0;
      int used = 0;

      // Clear the weight gradients once per batch; every image in the batch adds to them.
      memset(myGW1, 0, (size_t)MY_CONV1_WEIGHTS * 4);
      memset(myGB1, 0, CONV1_FILTERS * 4);
      memset(myGW2, 0, (size_t)MY_CONV2_WEIGHTS * 4);
      memset(myGB2, 0, CONV2_FILTERS * 4);
      memset(myGWD, 0, (size_t)MY_DENSE_WEIGHTS * 4);
      memset(myGBD, 0, NUM_CLASSES * 4);

      for (int i = batchStart; i < batchEnd; i++) {
        MyImageItem& img = myTrain[indices[i]];
        if (!myLoadImageFile(img.path.c_str())) continue;

        myForwardFloat();
        batchLoss += -log(max(myProbs[img.label], 1e-7f));
        if (myArgmaxProbs() == img.label) correctCount++;
        used++;

        myBackwardDense(img.label);
        myBackwardConv2();
        myBackwardPool1();
        myBackwardConv1();

        // Check for a stop request every few images so the exit feels instant.
        if (i % 3 == 0 && myTrainingExitRequested()) { myStopped = true; break; }
      }
      if (myStopped) break;
      if (used == 0) continue;

      myUpdateWeights(batch + 1);

      float avgLoss = batchLoss / used;
      float batchAcc = (float)correctCount / used;
      runningLoss += avgLoss;
      lossCount++;

      if ((batch + 1) % 5 == 0) {
        float displayLoss = runningLoss / lossCount;
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setCursor(0, 12); u8g2.print("Training...");
          u8g2.setCursor(0, 24);
          u8g2.print("B:"); u8g2.print(batch + 1);
          u8g2.print("/"); u8g2.print(totalBatches);
          u8g2.setCursor(0, 36);
          u8g2.print("L:"); u8g2.print(displayLoss, 3);
          u8g2.print(" A:"); u8g2.print((int)(batchAcc * 100)); u8g2.print("%");
        } while (u8g2.nextPage());
        runningLoss = 0;
        lossCount = 0;
      }

      if ((batch + 1) % 10 == 0) {
        Serial.printf("Batch %d/%d - Loss: %.4f - Acc: %.1f%%\n", batch + 1, totalBatches, avgLoss, batchAcc * 100);
      }
    }

    if (myStopped) {
      Serial.println("Training stopped - saving the weights so far");
      mySaveFloatWeights();
      myFloatLoaded = true;
      myInvalidateInt8();
      myResetMenuState();
      return;
    }

    Serial.println("\n--- Training Complete ---");

    // ---- Accuracy on the held-out validation images ----
    if (!myVal.empty()) {
      int valCorrect = 0, valCount = 0;
      for (auto& vitem : myVal) {
        if (!myLoadImageFile(vitem.path.c_str())) continue;
        myForwardFloat();
        if (myArgmaxProbs() == vitem.label) valCorrect++;
        valCount++;
      }
      if (valCount > 0) {
        Serial.printf("Validation Accuracy: %.1f%%  (%d/%d correct)\n", 100.0f * valCorrect / valCount, valCorrect, valCount);
      }
    }

    mySaveFloatWeights();
    myFloatLoaded = true;
    myInvalidateInt8();

    myShowMessage("DONE!", "Tap: again", "3+ taps: exit");
    myResetTouchState();

    Serial.println("Next: choose Convert to INT8 in the menu to make the INT8 model.");
    Serial.println("  Serial: t = train again   l = exit");
    Serial.println("  Touch:  1-2 taps = train again   3+ taps = exit");
    while (true) {
      if (Serial.available()) {
        char c = Serial.read();
        if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
          myResetMenuState();
          return;
        } else if (c == 't' || c == 'T') {
          break;
        }
      }
      int touchAction = myCheckTouchInput();
      if (touchAction == 2) {
        myResetMenuState();
        return;
      } else if (touchAction == 1) {
        Serial.println("Starting new training cycle");
        break;
      }
      delay(10);
    }
  }
}


// =============================================================================
//  CONVERT TO INT8  (menu item NUM_CLASSES+2)
//  Post-training quantization of the float model in memory. Uses the same math as the
//  web page: per-filter int8 weights, per-layer int8 activations, int32 biases and sums.
// =============================================================================

// Splits a real multiplier m into a 23-bit integer and a shift so that (acc * mult) >> shift is about acc * m.
void myQuantMult(double m, int32_t& mult, int32_t& shift) {
  if (!(m > 0)) { mult = 0; shift = 62; return; }
  int e;
  frexp(m, &e);                                        // m = f * 2^e with f in [0.5, 1), so e = floor(log2(m)) + 1
  long long mi = (long long)floor(m / ldexp(1.0, e) * 8388608.0 + 0.5);
  if (mi >= 8388608) { mi = 4194304; e++; }
  int sh = 23 - e;
  if (sh < 1) sh = 1;
  if (sh > 62) sh = 62;
  mult = (int32_t)mi;
  shift = sh;
}

// Turns a measured float range into an int8 scale and zero point (the range always includes 0).
void myActParams(double lo, double hi, double& scale, int32_t& zp) {
  if (lo > 0) lo = 0;
  if (hi < 0) hi = 0;
  if (hi - lo < 1e-6) hi = lo + 1e-6;
  scale = (hi - lo) / 255.0;
  double z = floor(-128.0 - lo / scale + 0.5);
  zp = (int32_t)(z < -128 ? -128 : (z > 127 ? 127 : z));
}

// Quantizes one layer. sx and zpx describe the layer input, os and ozp the layer output.
// worst collects the largest possible int32 sum so an overflow can be refused.
void myQuantLayer(const float* w, const float* b, int count, int per, double sx, int zpx, double os, int ozp, bool leaky, MyQLayer& L, double& worst) {
  for (int f = 0; f < count; f++) {
    double mx = 0;
    for (int k = 0; k < per; k++) {
      double a = fabs((double)w[f * per + k]);
      if (a > mx) mx = a;
    }
    double sw = (mx > 0 ? mx : 1e-8) / 127.0;
    long long sum = 0, absSum = 0;
    for (int k = 0; k < per; k++) {
      double v = floor((double)w[f * per + k] / sw + 0.5);
      if (v < -127) v = -127;
      if (v > 127) v = 127;
      L.w[f * per + k] = (int8_t)v;
      sum += (long long)v;
      absSum += (long long)fabs(v);
    }
    double bq = floor((double)b[f] / (sx * sw) + 0.5) - (double)zpx * (double)sum;
    if (bq > 2147483647.0) bq = 2147483647.0;
    if (bq < -2147483648.0) bq = -2147483648.0;
    L.bias[f] = (int32_t)bq;
    double M = sx * sw / os;
    myQuantMult(M, L.mp[f], L.sp[f]);
    myQuantMult(leaky ? 0.1 * M : M, L.mn[f], L.sn[f]);
    double w2 = (double)absSum * 128.0 + fabs(bq);
    if (w2 > worst) worst = w2;
  }
  L.zp = ozp;
}

// Writes /header/myWeightsInt8.bin (the same bytes the web page writes) and /header/myWeightsInt8.h.
bool mySaveInt8Files() {
  if (!mySDavailable) return false;
  if (!SD.exists("/header")) SD.mkdir("/header");
  if (SD.exists("/header/myWeightsInt8.bin")) SD.remove("/header/myWeightsInt8.bin");
  File f = SD.open("/header/myWeightsInt8.bin", FILE_WRITE);
  if (!f) {
    Serial.println("Could not open /header/myWeightsInt8.bin for writing");
    return false;
  }

  uint8_t head[32];
  memset(head, 0, sizeof(head));
  head[0] = 'W'; head[1] = 'I'; head[2] = '8'; head[3] = '1';
  uint16_t ver = 1, inSize = INPUT_SIZE;
  memcpy(head + 4, &ver, 2);
  memcpy(head + 6, &inSize, 2);
  head[8] = myChannels; head[9] = CONV1_FILTERS; head[10] = CONV2_FILTERS; head[11] = NUM_CLASSES;
  memcpy(head + 12, &myLogitScale, 4);
  memcpy(head + 16, &myLogitZp, 4);
  f.write(head, 32);

  MyQLayer* layers[3] = { &myL1, &myL2, &myLD };
  int wCounts[3] = { MY_CONV1_WEIGHTS, MY_CONV2_WEIGHTS, MY_DENSE_WEIGHTS };
  int counts[3] = { CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES };
  for (int i = 0; i < 3; i++) {
    MyQLayer& L = *layers[i];
    f.write((uint8_t*)L.w, (size_t)wCounts[i]);
    f.write((uint8_t*)L.bias, counts[i] * 4);
    f.write((uint8_t*)L.mp, counts[i] * 4);
    f.write((uint8_t*)L.sp, counts[i] * 4);
    f.write((uint8_t*)L.mn, counts[i] * 4);
    f.write((uint8_t*)L.sn, counts[i] * 4);
    f.write((uint8_t*)&L.zp, 4);
  }
  f.write(myTestImage, (size_t)INPUT_SIZE * INPUT_SIZE * myChannels);
  f.write((uint8_t*)myTestLogits, NUM_CLASSES);
  size_t total = f.size();
  f.close();
  if (total != myExpectedBlobBytes()) {
    Serial.printf("WARNING: wrote %u bytes but expected %u bytes\n", (unsigned)total, (unsigned)myExpectedBlobBytes());
    return false;
  }
  Serial.printf("Wrote /header/myWeightsInt8.bin (%u bytes)\n", (unsigned)total);

  // The same bytes as a C array, read back from the .bin in small pieces.
  if (SD.exists("/header/myWeightsInt8.h")) SD.remove("/header/myWeightsInt8.h");
  File h = SD.open("/header/myWeightsInt8.h", FILE_WRITE);
  File r = SD.open("/header/myWeightsInt8.bin", FILE_READ);
  if (!h || !r) {
    if (h) h.close();
    if (r) r.close();
    Serial.println("Could not write /header/myWeightsInt8.h");
    return true;
  }
  h.println("#ifndef MY_WEIGHTS_INT8_H\n#define MY_WEIGHTS_INT8_H");
  h.printf("// INPUT_SIZE %d, CONV1_FILTERS %d, CONV2_FILTERS %d, NUM_CLASSES %d, channels %d\n", INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES, myChannels);
  h.print("// Classes: ");
  for (int i = 0; i < NUM_CLASSES; i++) {
    h.print(myClassLabels[i]);
    if (i < NUM_CLASSES - 1) h.print(", ");
  }
  h.println("\nconst unsigned char myInt8Blob[] = {");
  uint8_t chunk[24];
  size_t left = total;
  while (left > 0) {
    size_t n = r.read(chunk, left < 24 ? left : 24);
    if (n == 0) break;
    for (size_t i = 0; i < n; i++) {
      h.print((int)chunk[i]);
      if (i + 1 < n || left > n) h.print(',');
    }
    h.println();
    left -= n;
  }
  h.println("};\n#endif");
  h.close();
  r.close();
  Serial.println("Wrote /header/myWeightsInt8.h");
  return true;
}

void myActionConvert() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot convert");
    myShowMessage("No SD card");
    delay(2000);
    myResetMenuState();
    return;
  }
  if (!myFloatLoaded) {
    Serial.println("ERROR: no float model in memory. Choose Train FLOAT, or put myWeights.bin on the SD card and reboot.");
    myShowMessage("No float", "model to", "convert");
    delay(3000);
    myResetMenuState();
    return;
  }

  Serial.println("\n>>> Convert to INT8   ('x' or 3+ taps = cancel)");
  myResetTouchState();
  myShowMessage("CONVERT INT8", "Loading...");

  std::vector<MyImageItem> myAll, myTrain, myVal;
  myBuildImageList(myAll, myTrain, myVal);
  if (myAll.empty()) {
    Serial.println("No images found in /images/<class>/");
    myShowMessage("No Images!");
    delay(2000);
    myResetMenuState();
    return;
  }

  // ---- Step 1: measure the value range of each layer on the training images ----
  std::vector<MyImageItem>& pool = myTrain.empty() ? myAll : myTrain;
  int want = CALIBRATION_IMAGES < 1 ? 1 : CALIBRATION_IMAGES;
  int gap = (int)pool.size() / want;
  if (gap < 1) gap = 1;
  double lo1 = 0, hi1 = 0, lo2 = 0, hi2 = 0, lo3 = 0, hi3 = 0;
  int used = 0;
  Serial.printf("Step 1 of 3: measuring ranges on up to %d images\n", want);
  for (int i = 0; i < (int)pool.size() && used < want; i += gap) {
    if (myTrainingExitRequested()) {
      Serial.println("Conversion cancelled. Nothing was changed.");
      myResetMenuState();
      return;
    }
    if (!myLoadImageFile(pool[i].path.c_str())) continue;
    myForwardFloat();
    for (int k = 0; k < CONV1_OUT_BYTES; k++) { double v = myFC1[k]; if (v < lo1) lo1 = v; if (v > hi1) hi1 = v; }
    for (int k = 0; k < FLATTENED_SIZE; k++)  { double v = myFC2[k]; if (v < lo2) lo2 = v; if (v > hi2) hi2 = v; }
    for (int k = 0; k < NUM_CLASSES; k++)     { double v = myFLogits[k]; if (v < lo3) lo3 = v; if (v > hi3) hi3 = v; }
    used++;
    if (used % 5 == 0) {
      u8g2.firstPage();
      do {
        u8g2.setFont(u8g2_font_5x7_tf);
        u8g2.setCursor(0, 12); u8g2.print("Converting...");
        u8g2.setCursor(0, 24); u8g2.print("Ranges ");
        u8g2.print(used); u8g2.print("/"); u8g2.print(want);
      } while (u8g2.nextPage());
    }
  }
  if (used == 0) {
    Serial.println("No image could be read. Nothing was changed.");
    myShowMessage("No images", "could be read");
    delay(2000);
    myResetMenuState();
    return;
  }
  Serial.printf("Used %d images. Ranges: conv1 %.2f to %.2f, conv2 %.2f to %.2f, logits %.2f to %.2f\n", used, lo1, hi1, lo2, hi2, lo3, hi3);

  // ---- Step 2: quantize the three layers (the INT8 model in memory is replaced) ----
  Serial.println("Step 2 of 3: quantizing the weights");
  myModelLoaded = false;
  myHasTest = false;
  double s1, s2, s3;
  int32_t z1, z2, z3;
  myActParams(lo1, hi1, s1, z1);
  myActParams(lo2, hi2, s2, z2);
  myActParams(lo3, hi3, s3, z3);
  double worst = 0;
  myQuantLayer(myFW1, myFB1, CONV1_FILTERS, 9 * myChannels, 1.0 / 255.0, -128, s1, z1, true, myL1, worst);
  myQuantLayer(myFW2, myFB2, CONV2_FILTERS, 9 * CONV1_FILTERS, s1, z1, s2, z2, true, myL2, worst);
  myQuantLayer(myFWD, myFBD, NUM_CLASSES, FLATTENED_SIZE, s2, z2, s3, z3, false, myLD, worst);
  Serial.printf("Largest possible int32 sum: %.0f of 2147483647\n", worst);
  if (worst > 2147483647.0) {
    Serial.println("REFUSED: an int32 sum could overflow with this layout. Use a smaller INPUT_SIZE or fewer filters.");
    myShowMessage("INT8 refused", "sums overflow");
    delay(3000);
    myResetMenuState();
    return;
  }
  myLogitScale = (float)s3;
  myLogitZp = z3;

  // ---- Step 3: a test image for the boot self-test, then save ----
  MyImageItem& testItem = myVal.empty() ? myAll[0] : myVal[0];
  if (!myLoadImageFile(testItem.path.c_str())) {
    Serial.println("Could not read the test image. Nothing was saved.");
    myResetMenuState();
    return;
  }
  int myTotalIn = INPUT_SIZE * INPUT_SIZE * myChannels;
  for (int i = 0; i < myTotalIn; i++) myTestImage[i] = (uint8_t)((int)myIn[i] + 128);
  myForwardInt8();
  for (int c = 0; c < NUM_CLASSES; c++) myTestLogits[c] = myLogitsQ[c];
  myHasTest = true;
  myModelLoaded = true;
  Serial.println("Step 3 of 3: saving to the SD card");
  bool saved = mySaveInt8Files();

  // ---- Compare float and INT8 on the validation images (all images when there are none) ----
  std::vector<MyImageItem>& evalSet = myVal.empty() ? myAll : myVal;
  int fOk = 0, qOk = 0, same = 0, n = 0;
  float maxDiff = 0;
  unsigned long fUs = 0, qUs = 0;
  for (auto& item : evalSet) {
    if (!myLoadImageFile(item.path.c_str())) continue;
    unsigned long t0 = micros();
    myForwardFloat();
    fUs += micros() - t0;
    float pf[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) pf[c] = myProbs[c];
    int a = myArgmaxProbs();
    t0 = micros();
    myForwardInt8();
    qUs += micros() - t0;
    int b = myArgmaxProbs();
    if (a == item.label) fOk++;
    if (b == item.label) qOk++;
    if (a == b) same++;
    for (int c = 0; c < NUM_CLASSES; c++) {
      float d = fabs(pf[c] - myProbs[c]) * 100.0f;
      if (d > maxDiff) maxDiff = d;
    }
    n++;
  }
  Serial.println("\n--- INT8 conversion complete ---");
  if (n > 0) {
    Serial.printf("On %d %s images: float %.1f%%, INT8 %.1f%%, same answer %.1f%%, largest probability difference %.2f points\n",
                  n, myVal.empty() ? "all" : "validation", 100.0f * fOk / n, 100.0f * qOk / n, 100.0f * same / n, maxDiff);
    Serial.printf("Network time per image: float %lu us, INT8 %lu us\n", fUs / n, qUs / n);
  }
  Serial.println(saved ? "INT8 model is ready: choose Infer INT8." : "WARNING: the INT8 model is in memory but the SD card files were not saved.");
  myShowMessage(saved ? "INT8 saved" : "Save failed", n > 0 ? "see serial log" : "");
  delay(2500);
  myResetMenuState();
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
//    @F <kind> <n> <pred> <probs|-> <logits|-> <layout> <centre input|-> <heatSide> <heat base64|-> <jpeg base64> <model I8 or F32>
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
    for (int i = 0; i < NUM_CLASSES; i++) {
      if (i) Serial.print(',');
      if (myActiveEngine == 1) Serial.print(myFLogits[i], 4);
      else                     Serial.print(((int)myLogitsQ[i] - myLogitZp) * myLogitScale, 4);
    }
  } else {
    Serial.print("- -");
  }
  Serial.printf(" %dx%dx%dx%d ", INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, myChannels);
  if (inf) {
    int c = ((INPUT_SIZE / 2) * INPUT_SIZE + INPUT_SIZE / 2) * myChannels;
    for (int k = 0; k < myChannels; k++) {
      if (k) Serial.print(',');
      Serial.print(myActiveEngine == 1 ? myInF[c + k] : ((int)myIn[c + k] + 128) / 255.0f, 4);
    }
  } else {
    Serial.print('-');
  }
  Serial.print(' ');
  if (inf) {
    const int hn = CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    static uint8_t heat[CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE];
    float lo = 1e30f, hi = -1e30f;
    for (int i = 0; i < hn; i++) {
      float m = myHeatVal(i);
      for (int f = 1; f < CONV2_FILTERS; f++) { float v = myHeatVal(f * hn + i); if (v > m) m = v; }
      if (m < lo) lo = m;
      if (m > hi) hi = m;
    }
    float span = hi - lo;
    if (span < 1e-9f) span = 1.0f;
    for (int i = 0; i < hn; i++) {
      float m = myHeatVal(i);
      for (int f = 1; f < CONV2_FILTERS; f++) { float v = myHeatVal(f * hn + i); if (v > m) m = v; }
      heat[i] = (uint8_t)(255.0f * (m - lo) / span + 0.5f);
    }
    Serial.printf("%d ", CONV2_OUTPUT_SIZE);
    myPrintB64(heat, hn);
  } else {
    Serial.print("0 -");
  }
  Serial.print(' ');
  myPrintB64(fb->buf, fb->len);
  Serial.print(myActiveEngine == 1 ? " F32" : " I8");
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
//  THE FLOAT CNN  (same network as the INT8 one, on floats)
//    input INPUT_SIZE x INPUT_SIZE x channels (pixel / 255)
//    -> conv 3x3 -> leaky ReLU (0.1) -> 2x2 max pool -> conv 3x3 -> leaky ReLU
//    -> flatten -> dense -> softmax.   Sums are clipped to -100..100, scores to -50..50.
// =============================================================================

// Turns the INT8 input (pixel - 128) into the float input (pixel / 255), so both models see the same picture.
void myFillFloatInput() {
  int total = INPUT_SIZE * INPUT_SIZE * myChannels;
  for (int i = 0; i < total; i++) myInF[i] = ((int)myIn[i] + 128) * 0.003921569f;
}

// Value of the conv2 map used for the web page heatmap, for whichever model is active.
float myHeatVal(int i) {
  return myActiveEngine == 1 ? myFC2[i] : (float)myC2[i];
}

int myArgmaxProbs() {
  int best = 0;
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (myProbs[i] > myProbs[best]) best = i;
  }
  return best;
}

void myForwardFloat() {
  const int ch = myChannels;
  const int stride1 = 9 * ch;

  // Conv1
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    for (int y = 0; y < CONV1_OUTPUT_SIZE; y++) {
      for (int x = 0; x < CONV1_OUTPUT_SIZE; x++) {
        float a = 0;
        for (int ky = 0; ky < 3; ky++) {
          for (int kx = 0; kx < 3; kx++) {
            const float* ip = myInF + ((y + ky) * INPUT_SIZE + (x + kx)) * ch;
            const float* wp = myFW1 + f * stride1 + (ky * 3 + kx) * ch;
            if (ch == 3) a += ip[0] * wp[0] + ip[1] * wp[1] + ip[2] * wp[2];
            else         a += ip[0] * wp[0];
          }
        }
        a += myFB1[f];
        a = a < -100.0f ? -100.0f : (a > 100.0f ? 100.0f : a);
        myFC1[ob + y * CONV1_OUTPUT_SIZE + x] = a > 0 ? a : 0.1f * a;
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
        float m = myFC1[i0];
        if (myFC1[i0 + 1] > m) m = myFC1[i0 + 1];
        if (myFC1[i2] > m) m = myFC1[i2];
        if (myFC1[i2 + 1] > m) m = myFC1[i2 + 1];
        myFP1[ob + y * POOL1_OUTPUT_SIZE + x] = m;
      }
    }
  }

  // Conv2
  for (int f = 0; f < CONV2_FILTERS; f++) {
    int ob = f * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    for (int y = 0; y < CONV2_OUTPUT_SIZE; y++) {
      for (int x = 0; x < CONV2_OUTPUT_SIZE; x++) {
        float a = 0;
        for (int c = 0; c < CONV1_FILTERS; c++) {
          int ib = c * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
          for (int ky = 0; ky < 3; ky++) {
            const float* ip = myFP1 + ib + (y + ky) * POOL1_OUTPUT_SIZE + x;
            const float* wp = myFW2 + f * 9 * CONV1_FILTERS + c * 9 + ky * 3;
            a += ip[0] * wp[0] + ip[1] * wp[1] + ip[2] * wp[2];
          }
        }
        a += myFB2[f];
        a = a < -100.0f ? -100.0f : (a > 100.0f ? 100.0f : a);
        myFC2[ob + y * CONV2_OUTPUT_SIZE + x] = a > 0 ? a : 0.1f * a;
      }
    }
  }

  // Dense layer (a double sum keeps the long sum accurate)
  float mx = -1e30f;
  for (int c = 0; c < NUM_CLASSES; c++) {
    double sum = 0;
    const float* wp = myFWD + (size_t)c * FLATTENED_SIZE;
    for (int i = 0; i < FLATTENED_SIZE; i++) sum += (double)myFC2[i] * wp[i];
    float v = (float)sum + myFBD[c];
    v = v < -50.0f ? -50.0f : (v > 50.0f ? 50.0f : v);
    myFLogits[c] = v;
    if (v > mx) mx = v;
  }

  // Softmax
  float expSum = 0;
  for (int c = 0; c < NUM_CLASSES; c++) expSum += exp(myFLogits[c] - mx);
  for (int c = 0; c < NUM_CLASSES; c++) myProbs[c] = exp(myFLogits[c] - mx) / expSum;
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
//  MENU ITEMS NUM_CLASSES+1 and NUM_CLASSES+2 : INFER (LIVE CLASSIFICATION)
//  engine 0 = INT8 model, engine 1 = float model.
//  Every camera frame is shrunk to INPUT_SIZE and run through the chosen network.
//  Serial prints the whole frame time and the network-only time in microseconds.
//  Serial 's' switches to the other model without leaving. Exit with 't' or 'l', or touch the pad.
// =============================================================================
void myActionInfer(int engine) {
  const char* myEngineName = engine == 1 ? "FLOAT" : "INT8";
  bool myReady = engine == 1 ? myFloatLoaded : myModelLoaded;
  if (!myReady) {
    Serial.printf("ERROR: no %s model in memory.\n", myEngineName);
    if (engine == 1) Serial.println("Put myWeights.bin from the web page on the SD card and reboot. If the boot log said REFUSED, make the settings match and reflash.");
    else             Serial.println("Put myWeightsInt8.bin from the web page on the SD card and reboot, or bake in myWeightsInt8.h. If the boot log said REFUSED, make the settings match and reflash.");
    myShowMessage("No model!", engine == 1 ? "Save float on" : "Convert on the", engine == 1 ? "the web page" : "web page first");
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

  myActiveEngine = engine;
  Serial.printf("\n>>> Inference mode (%s)\n", myEngineName);
  Serial.println("  Serial 's' = switch model, 't' or 'l' = exit, or touch the pad");

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
      } else if (c == 's' || c == 'S') {
        bool otherReady = (myActiveEngine == 1) ? myModelLoaded : myFloatLoaded;
        if (otherReady) {
          myActiveEngine = 1 - myActiveEngine;
          myEngineName = myActiveEngine == 1 ? "FLOAT" : "INT8";
          Serial.printf("Switched to the %s model\n", myEngineName);
        } else {
          Serial.println("The other model is not loaded");
        }
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
      if (myActiveEngine == 1) myFillFloatInput();

      unsigned long netStart = micros();
      if (myActiveEngine == 1) myForwardFloat();
      else                     myForwardInt8();
      netUs = micros() - netStart;
      myInferCount++;

      pred = myArgmaxProbs();

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
          u8g2.drawBox(oW - 18, 0, 18, 8);
          u8g2.setColorIndex(1);
          char buf[20];
          snprintf(buf, sizeof(buf), "%s %d%%", myClassLabels[pred].c_str(), (int)(myProbs[pred] * 100));
          u8g2.drawStr(1, oH - 1, buf);
          u8g2.drawStr(oW - 17, 7, myActiveEngine == 1 ? "F32" : "I8");
        } while (u8g2.nextPage());
      }
    }

    esp_camera_fb_return(fb);

    unsigned long frameMs = millis() - frameStart;
    Serial.printf("[%s] Frame %d: %lu ms (%.1f FPS) network %lu us Current Pred: %s (%.1f%%) | All:",
                  myActiveEngine == 1 ? "FLOAT" : "INT8", frameIndex + 1, frameMs, 1000.0 / max(frameMs, 1UL), netUs,
                  myClassLabels[pred].c_str(), myProbs[pred] * 100);
    for (int i = 0; i < NUM_CLASSES; i++) Serial.printf(" %.0f%%", myProbs[i] * 100);
    Serial.println();

    frameIndex++;
    if (frameIndex >= 10) frameIndex = 0;
  }
}


// =============================================================================
//  MENU
//  Items 1..NUM_CLASSES collect images, then Infer INT8, then Infer FLOAT.
//  Called from loop().
// =============================================================================

// =============================================================================
//  BENCH MOBILENET  (INT8, synthetic weights: measures time and memory only)
//  mode 0 = scalar loops, 1 = fast loops, 2 = fast loops on both CPU cores.
//  All modes use the same arithmetic, so their output checksums must match.
// =============================================================================
#if MY_BENCH_O3
#pragma GCC push_options
#pragma GCC optimize ("O3")
#endif
int8_t myBenchClamp(int32_t acc) {
  acc >>= 7;
  return acc < -128 ? -128 : (acc > 127 ? 127 : acc);
}

void myBenchFirstConvRows(const int8_t* in, int8_t* out, const int8_t* w, int size, int y0, int y1) {
  int outSize = size / 2;
  for (int y = y0; y < y1; y++) for (int x = 0; x < outSize; x++) for (int f = 0; f < 8; f++) {
    int32_t acc = 0;
    for (int ky = 0; ky < 3; ky++) {
      int iy = y * 2 + ky - 1;
      if (iy < 0 || iy >= size) continue;
      for (int kx = 0; kx < 3; kx++) {
        int ix = x * 2 + kx - 1;
        if (ix < 0 || ix >= size) continue;
        const int8_t* p = in + (iy * size + ix) * 3;
        const int8_t* q = w + f * 27 + (ky * 3 + kx) * 3;
        acc += p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
      }
    }
    out[(y * outSize + x) * 8 + f] = myBenchClamp(acc);
  }
}

void myBenchFirstConv(const int8_t* in, int8_t* out, const int8_t* w, int size) {
  myBenchFirstConvRows(in, out, w, size, 0, size / 2);
}

void myBenchDepthwise(const int8_t* in, int8_t* out, const int8_t* w, int size, int ch, int stride) {
  int outSize = stride == 1 ? size : size / 2;
  for (int y = 0; y < outSize; y++) for (int x = 0; x < outSize; x++) for (int c = 0; c < ch; c++) {
    int32_t acc = 0;
    for (int ky = 0; ky < 3; ky++) {
      int iy = y * stride + ky - 1;
      if (iy < 0 || iy >= size) continue;
      for (int kx = 0; kx < 3; kx++) {
        int ix = x * stride + kx - 1;
        if (ix < 0 || ix >= size) continue;
        acc += in[(iy * size + ix) * ch + c] * w[(ky * 3 + kx) * ch + c];
      }
    }
    out[(y * outSize + x) * ch + c] = myBenchClamp(acc);
  }
}

void myBenchPointwise(const int8_t* in, int8_t* out, const int8_t* w, int pixels, int inCh, int outCh) {
  for (int p = 0; p < pixels; p++) {
    const int8_t* src = in + (size_t)p * inCh;
    for (int o = 0; o < outCh; o++) {
      const int8_t* wf = w + (size_t)o * inCh;
      int32_t acc = 0;
      for (int i = 0; i < inCh; i++) acc += src[i] * wf[i];
      out[(size_t)p * outCh + o] = myBenchClamp(acc);
    }
  }
}

// Fast depthwise: the border test is done once per tap, not once per channel, and channels are the inner loop.
void myBenchDepthwiseFast(const int8_t* in, int8_t* out, const int8_t* w, int size, int ch, int stride, int y0, int y1) {
  int outSize = stride == 1 ? size : size / 2;
  int32_t acc[256];
  for (int y = y0; y < y1; y++) for (int x = 0; x < outSize; x++) {
    for (int c = 0; c < ch; c++) acc[c] = 0;
    for (int ky = 0; ky < 3; ky++) {
      int iy = y * stride + ky - 1;
      if (iy < 0 || iy >= size) continue;
      for (int kx = 0; kx < 3; kx++) {
        int ix = x * stride + kx - 1;
        if (ix < 0 || ix >= size) continue;
        const int8_t* p = in + (iy * size + ix) * ch;
        const int8_t* q = w + (ky * 3 + kx) * ch;
        for (int c = 0; c < ch; c++) acc[c] += p[c] * q[c];
      }
    }
    int8_t* o = out + (y * outSize + x) * ch;
    for (int c = 0; c < ch; c++) o[c] = myBenchClamp(acc[c]);
  }
}

// Fast pointwise: 2 pixels x 4 output channels per pass, so each loaded value is used several times.
void myBenchPointwiseFast(const int8_t* in, int8_t* out, const int8_t* w, int inCh, int outCh, int p0, int p1) {
  int p = p0;
  for (; p + 1 < p1; p += 2) {
    const int8_t* s0 = in + (size_t)p * inCh;
    const int8_t* s1 = s0 + inCh;
    int8_t* d0 = out + (size_t)p * outCh;
    int8_t* d1 = d0 + outCh;
    for (int o = 0; o + 3 < outCh; o += 4) {
      const int8_t* w0 = w + (size_t)o * inCh;
      const int8_t* w1 = w0 + inCh;
      const int8_t* w2 = w1 + inCh;
      const int8_t* w3 = w2 + inCh;
      int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
      for (int i = 0; i < inCh; i++) {
        int x0 = s0[i], x1 = s1[i];
        int q0 = w0[i], q1 = w1[i], q2 = w2[i], q3 = w3[i];
        a0 += x0 * q0; a1 += x0 * q1; a2 += x0 * q2; a3 += x0 * q3;
        b0 += x1 * q0; b1 += x1 * q1; b2 += x1 * q2; b3 += x1 * q3;
      }
      d0[o] = myBenchClamp(a0); d0[o + 1] = myBenchClamp(a1); d0[o + 2] = myBenchClamp(a2); d0[o + 3] = myBenchClamp(a3);
      d1[o] = myBenchClamp(b0); d1[o + 1] = myBenchClamp(b1); d1[o + 2] = myBenchClamp(b2); d1[o + 3] = myBenchClamp(b3);
    }
  }
  for (; p < p1; p++) {
    const int8_t* s0 = in + (size_t)p * inCh;
    int8_t* d0 = out + (size_t)p * outCh;
    for (int o = 0; o < outCh; o++) {
      const int8_t* w0 = w + (size_t)o * inCh;
      int32_t a0 = 0;
      for (int i = 0; i < inCh; i++) a0 += s0[i] * w0[i];
      d0[o] = myBenchClamp(a0);
    }
  }
}

// kind 0 = first conv (rows), 1 = depthwise (rows), 2 = pointwise (pixels)
void myBenchDoJob(const MyBenchJob* j) {
  if (j->kind == 0) myBenchFirstConvRows(j->in, j->out, j->w, j->size, j->from, j->to);
  else if (j->kind == 1) myBenchDepthwiseFast(j->in, j->out, j->w, j->size, j->ch, j->stride, j->from, j->to);
  else myBenchPointwiseFast(j->in, j->out, j->w, j->inCh, j->outCh, j->from, j->to);
}

void myBenchWorkerTask(void* arg) {
  while (true) {
    xSemaphoreTake(myBenchStartSem, portMAX_DELAY);
    if (myBenchWorkerQuit) break;
    myBenchDoJob(&myBenchJob);
    xSemaphoreGive(myBenchDoneSem);
  }
  xSemaphoreGive(myBenchDoneSem);
  vTaskDelete(NULL);
}

// Runs one layer. With dual = true the second half of the rows or pixels goes to the worker task on the other core.
void myBenchLayer(MyBenchJob job, int total, bool dual) {
  if (!dual || total < 2) {
    job.from = 0; job.to = total;
    myBenchDoJob(&job);
    return;
  }
  int mid = total / 2;
  myBenchJob = job;
  myBenchJob.from = mid; myBenchJob.to = total;
  xSemaphoreGive(myBenchStartSem);
  job.from = 0; job.to = mid;
  myBenchDoJob(&job);
  xSemaphoreTake(myBenchDoneSem, portMAX_DELAY);
}

#if MY_BENCH_O3
#pragma GCC pop_options
#endif
size_t myBenchWeightTotal() {
  size_t total = 216;
  int ch = 8;
  for (int b = 0; b < 13; b++) {
    total += 9 * ch + (size_t)ch * myBenchBlocks[b].outCh;
    ch = myBenchBlocks[b].outCh;
  }
  return total;
}

uint32_t myBenchSum(const int8_t* p, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; i++) h = (h ^ (uint8_t)p[i]) * 16777619u;
  return h;
}

// Runs the whole backbone once (each layer reads its own slice of the weight buffer). Returns the multiply-accumulate count.
double myBenchRun(int8_t* myA, int8_t* myB, const int8_t* myW, int res, int mode) {
  bool fast = mode > 0, dual = mode == 2;
  int8_t* myIn = myA;
  int8_t* myOut = myB;
  const int8_t* wp = myW;
  int size = res / 2, ch = 8;
  double macs = (double)size * size * 8 * 27;
  if (fast) myBenchLayer({0, myIn, myOut, wp, res, 8, 2, 3, 8, 0, 0}, size, dual);
  else myBenchFirstConv(myIn, myOut, wp, res);
  wp += 216;
  for (int b = 0; b < 13; b++) {
    int outCh = myBenchBlocks[b].outCh, stride = myBenchBlocks[b].stride;
    int outSize = stride == 1 ? size : size / 2;
    int8_t* t = myIn; myIn = myOut; myOut = t;
    const int8_t* dw = wp;
    wp += 9 * ch;
    if (fast) myBenchLayer({1, myIn, myOut, dw, size, ch, stride, ch, ch, 0, 0}, outSize, dual);
    else myBenchDepthwise(myIn, myOut, dw, size, ch, stride);
    size = outSize;
    macs += (double)size * size * ch * 9;
    t = myIn; myIn = myOut; myOut = t;
    const int8_t* pw = wp;
    wp += (size_t)ch * outCh;
    if (fast) myBenchLayer({2, myIn, myOut, pw, size, ch, 1, ch, outCh, 0, 0}, size * size, dual);
    else myBenchPointwise(myIn, myOut, pw, size * size, ch, outCh);
    macs += (double)size * size * ch * outCh;
    ch = outCh;
  }
  myBenchOutPtr = myOut;
  myBenchOutLen = (size_t)size * size * ch;
  return macs;
}

void myActionBenchMobileNet() {
  Serial.println("\n>>> Bench MobileNet   ('x' or 3+ taps = cancel)");
  Serial.println("MobileNetV1 alpha 0.25 layer shapes, INT8, synthetic weights. Measures time and memory only; it does not classify.");
  Serial.printf("Kernel optimization: %s\n", myBenchO3 ? "-O3" : "build default");
  Serial.println("Every variant must print the same checksum for a given size, which proves the faster loops compute the same result.");
  myResetTouchState();
  myShowMessage("MOBILENET", "BENCH", "Running...");
  size_t myWeightBytes = myBenchWeightTotal();
  myBenchStartSem = xSemaphoreCreateBinary();
  myBenchDoneSem = xSemaphoreCreateBinary();
  myBenchWorkerQuit = false;
  bool myDualOk = myBenchStartSem && myBenchDoneSem &&
                  xTaskCreatePinnedToCore(myBenchWorkerTask, "myBench", 6144, NULL, 1, &myBenchWorker, 0) == pdPASS;
  Serial.printf("Backbone weights: %u bytes. Free internal RAM %u (largest block %u), free PSRAM %u. Second core: %s\n",
                (unsigned)myWeightBytes, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL), (unsigned)ESP.getFreePsram(), myDualOk ? "ready" : "not available");
  bool myCancelled = false, myMismatch = false;
  char myLine2[24];
  char myLine3[24];
  for (int s = 0; s < 2 && !myCancelled; s++) {
    int myRes = myBenchSizes[s];
    size_t myActBytes = (size_t)(myRes / 2) * (myRes / 2) * 16, myInBytes = (size_t)myRes * myRes * 3;
    uint32_t myRef = 0;
    bool myHaveRef = false;
    double myBestMs = 1e9;
    Serial.printf("--- input %dx%dx3 ---\n", myRes, myRes);
    for (int cfg = 0; cfg < 3 && !myCancelled; cfg++) {
      uint32_t myActCaps = (cfg == 0 ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL) | MALLOC_CAP_8BIT;
      uint32_t myWeightCaps = (cfg == 2 ? MALLOC_CAP_INTERNAL : MALLOC_CAP_SPIRAM) | MALLOC_CAP_8BIT;
      int8_t* myA = (int8_t*)heap_caps_malloc(myActBytes, myActCaps);
      int8_t* myB = (int8_t*)heap_caps_malloc(myActBytes, myActCaps);
      int8_t* myW = (int8_t*)heap_caps_malloc(myWeightBytes, myWeightCaps);
      if (!myA || !myB || !myW) {
        Serial.printf("  %s: skipped, not enough memory (needs %u + %u activation bytes and %u weight bytes)\n", myBenchConfigNames[cfg], (unsigned)myActBytes, (unsigned)myActBytes, (unsigned)myWeightBytes);
        heap_caps_free(myA); heap_caps_free(myB); heap_caps_free(myW);
        continue;
      }
      for (size_t i = 0; i < myWeightBytes; i++) myW[i] = (int8_t)((i * 53 + 7) % 255 - 127);
      for (int mode = 0; mode < 3 && !myCancelled; mode++) {
        if (mode == 0 && cfg > 0) continue;
        if (mode == 2 && !myDualOk) continue;
        unsigned long myBest = 0xFFFFFFFFUL;
        double myMacs = 0;
        uint32_t mySum = 0;
        for (int p = 0; p < myBenchPasses; p++) {
          if (myTrainingExitRequested()) { myCancelled = true; break; }
          for (size_t i = 0; i < myInBytes; i++) myA[i] = (int8_t)(i * 37 + 11);
          unsigned long myStart = micros();
          myMacs = myBenchRun(myA, myB, myW, myRes, mode);
          unsigned long myTime = micros() - myStart;
          if (myTime < myBest) myBest = myTime;
          mySum = myBenchSum(myBenchOutPtr, myBenchOutLen);
        }
        if (myCancelled) break;
        if (!myHaveRef) { myRef = mySum; myHaveRef = true; }
        bool myMatch = mySum == myRef;
        if (!myMatch) myMismatch = true;
        double myMs = myBest / 1000.0;
        if (myMs < myBestMs) myBestMs = myMs;
        Serial.printf("  %-32s %-15s %7.0f ms  %5.2f per second  %6.1f million MACs/s  checksum %08X %s\n",
                      myBenchConfigNames[cfg], myBenchModeNames[mode], myMs, 1000.0 / myMs, myMacs / (myBest > 0 ? myBest : 1), (unsigned)mySum, myMatch ? "ok" : "MISMATCH");
      }
      heap_caps_free(myA); heap_caps_free(myB); heap_caps_free(myW);
    }
    if (myCancelled) break;
    snprintf(myLine2, sizeof(myLine2), "%dx%d", myRes, myRes);
    snprintf(myLine3, sizeof(myLine3), "best %.0f ms", myBestMs);
    myShowMessage("BENCH", myLine2, myLine3);
    delay(1200);
  }
  if (myDualOk) {
    myBenchWorkerQuit = true;
    xSemaphoreGive(myBenchStartSem);
    xSemaphoreTake(myBenchDoneSem, 1000 / portTICK_PERIOD_MS);
    delay(20);
  }
  if (myBenchStartSem) vSemaphoreDelete(myBenchStartSem);
  if (myBenchDoneSem) vSemaphoreDelete(myBenchDoneSem);
  myBenchStartSem = NULL;
  myBenchDoneSem = NULL;
  Serial.println(myCancelled ? "Bench cancelled" : (myMismatch ? "Bench done. WARNING: a checksum did not match." : "Bench done. All checksums match."));
  delay(2000);
  myResetMenuState();
}

// Text for menu item 1..myTotalItems.
String myMenuLabel(int item) {
  if (item <= NUM_CLASSES) return myClassLabels[item - 1];
  if (item == NUM_CLASSES + 1) return "Train FLOAT";
  if (item == NUM_CLASSES + 2) return myFloatLoaded ? "Convert to INT8" : "Convert (no float)";
  if (item == NUM_CLASSES + 3) return myModelLoaded ? "Infer INT8" : "Infer INT8 (none)";
  if (item == NUM_CLASSES + 4) return myFloatLoaded ? "Infer FLOAT" : "Infer FLOAT (none)";
  return "Bench MobileNet";
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
  if (idx <= NUM_CLASSES)             myActionCollect(idx - 1);
  else if (idx == NUM_CLASSES + 1)    myActionTrain();
  else if (idx == NUM_CLASSES + 2)    myActionConvert();
  else if (idx == NUM_CLASSES + 3)    myActionInfer(0);
  else if (idx == NUM_CLASSES + 4)    myActionInfer(1);
  else                                myActionBenchMobileNet();
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
