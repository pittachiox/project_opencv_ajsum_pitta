import re

with open('c:/Users/HP/source/repos/final/online1.h', 'r', encoding='utf-8') as f:
    content = f.read()

# Step 1: Replace global variables declarations with member variables
# Pattern: __declspec(selectany) <type> <name> [= <default>];
content = re.sub(
    r'__declspec\(selectany\)\s+(.+?)\s+([a-zA-Z0-9_]+)(\s*=\s*[^;]+)?\s*;',
    r'\1 \2\3;',
    content
)

# Fix atomic initializers (e.g., std::atomic<int> g_droppedFrames_online(0); -> {0};)
content = re.sub(
    r'(std::atomic<.+?>\s+[a-zA-Z0-9_]+)\((.*?)\);',
    r'\1{\2};',
    content
)

# Step 2: Remove 'static ' from function definitions to make them member methods
content = re.sub(r'^static\s+', '', content, flags=re.MULTILINE)

# We need to wrap everything from line 106 to the end inside `class CameraInstance { public: ... };`
# Let's find exactly where the class would start.
# In the file, there is 'class CameraInstance {' around line 96.
# Let's replace the existing stub with the real one.

stub_regex = re.compile(r'class CameraInstance\s*\{.*?\};', re.DOTALL)
content = stub_regex.sub('// CAMERA INSTANCE CLASS DEFINITION REPLACED', content)

# Now, we find the first global variable which we just replaced.
# It used to be `OnlineAppState g_onlineState;`
# We will insert the class declaration right before it.

first_var = "OnlineAppState g_onlineState;"
class_start = """
class CameraInstance {
public:
    int camera_id = 0;
    CameraConfig config;
    
    CameraInstance(const CameraConfig& cfg) : config(cfg) {
        camera_id = cfg.id;
    }
    
    ~CameraInstance() {
        StopCameraHeadless();
    }

"""
content = content.replace(first_var, class_start + first_var)

# And add the closing brace at the very end of the file.
content += "\n};\n"

with open('c:/Users/HP/source/repos/final/online1.h', 'w', encoding='utf-8') as f:
    f.write(content)

print("online1.h refactored successfully.")
