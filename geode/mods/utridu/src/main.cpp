#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <cocos2d.h>
#include <windows.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>

using namespace geode::prelude;

// --- SHARED MEMORY CONFIGURATION ---
const int MAX_OBJECTS = 30;
const char* MEM_NAME = "GD_RL_Memory";

struct ObjectData {
    float dx;
    float dy;
    float w;
    float h;
    int type; // 1=Spike, 2=Block, 5=Portal, -1=Empty
};

struct SharedState {
    // --- SYNCHRONIZATION ---
    volatile int cpp_writing;    // 1 when C++ is writing
    volatile int py_writing;     // 1 when Python is reading

    // --- PLAYER STATE ---
    float player_x;
    float player_y;
    float player_vel_x;          // Real calculated velocity (delta x / dt)
    float player_vel_y;
    float player_rot;
    int gravity;
    int is_on_ground;            // Changed from bool to int for alignment
    int is_dead;                 // Changed from bool to int for alignment
    int is_terminal;             // Changed from bool to int for alignment

    // --- REWARD SHAPING DATA ---
    float percent;
    float dist_nearest_hazard;   // NEW: Distance to closest spike
    float dist_nearest_solid;    // NEW: Distance to closest block
    int player_mode;
    float player_speed;          // Internal speed setting

    // --- ENVIRONMENT ---
    ObjectData objects[MAX_OBJECTS];

    // --- COMMANDS ---
    int action_command;
    int reset_command;
    int checkpoint_command;
};

// Global Handles
HANDLE hMapFile = NULL;
SharedState* pSharedMem = nullptr;


class $modify(MyPlayLayer, PlayLayer) {
    struct Fields {
        CCLabelTTF* m_statusLabel = nullptr;
        CCDrawNode* m_debugDrawNode = nullptr;
        bool m_isHolding = false;
        bool m_showDebug = true; 
        
        // NEW: Physics & Death Logic Fields
        float m_lastX = 0.0f;
        float m_lastPercent = 0.0f;
        int m_stuckFrames = 0;
    };

    // --- HELPER: GET CATEGORY STRING ---
    static std::string getObjectCategory(GameObject* go) {
        if (!go) return "unknown";
        int id = go->m_objectID;

        if (go->m_objectType == GameObjectType::Hazard) return "spike";
        if (go->m_objectType == GameObjectType::Solid) return "solid_block";

        if (id == 12) return "cube_portal";
        if (id == 13) return "ship_portal";
        if (id == 47) return "ball_portal";
        if (id == 111) return "ufo_portal";
        if (id == 660) return "wave_portal";
        if (id == 1331) return "spider_portal";
        if (id == 101) return "mini_portal";
        if (id == 99) return "normal_portal";

        return "decoration";
    }

    // --- HELPER: MAP CATEGORY TO INT ---
    int categoryToInt(const std::string& cat) {
        if (cat == "spike") return 1;
        if (cat == "solid_block") return 2;
        if (cat.find("portal") != std::string::npos) return 5;
        return 0;
    }

    struct RLObject {
        std::string category;
        CCRect rect;
        float dx, dy, w, h;
    };

    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;

        // --- DISABLE AUTO-CHECKPOINTS ---
        GameManager::sharedState()->setGameVariable("0027", false);

        auto winSize = CCDirector::sharedDirector()->getWinSize();

        // 1. SETUP SHARED MEMORY
        hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(SharedState), MEM_NAME);

        std::string statusMsg = "RL: Memory Ready (Press M to Toggle HUD)";
        if (hMapFile == NULL) {
            statusMsg = "ERR: CreateFile Failed! (Run Admin)";
        } else {
            pSharedMem = (SharedState*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedState));
            if (!pSharedMem) statusMsg = "ERR: MapView Failed!";
            else memset(pSharedMem, 0, sizeof(SharedState));
        }

        // 2. SETUP DRAW NODE
        auto drawNode = CCDrawNode::create();
        drawNode->setZOrder(99999);
        this->m_objectLayer->addChild(drawNode);
        m_fields->m_debugDrawNode = drawNode;

        // 3. SETUP HUD
        auto label = CCLabelTTF::create(statusMsg.c_str(), "Courier New", 24.0f);
        label->setScale(0.35f);
        label->setAnchorPoint({0.0f, 1.0f});
        label->setPosition(ccp(5.0f, winSize.height - 5.0f));
        label->enableStroke({0,0,0}, 2.0f);
        label->setColor({0, 255, 0});
        label->setHorizontalAlignment(kCCTextAlignmentLeft);

        if (this->m_uiLayer) this->m_uiLayer->addChild(label, 999);
        else this->addChild(label, 9999);

        m_fields->m_statusLabel = label;
        m_fields->m_isHolding = false;
        m_fields->m_showDebug = true;

        // Reset Physics Fields
        m_fields->m_lastX = 0.0f;
        m_fields->m_lastPercent = 0.0f;
        m_fields->m_stuckFrames = 0;

        // FORCE SCHEDULE LOOP
        this->schedule(schedule_selector(MyPlayLayer::rl_loop));

        return true;
    }

    // --- INPUT HOOK ---
    void keyDown(enumKeyCodes key) {
        if (key == enumKeyCodes::KEY_M) {
            m_fields->m_showDebug = !m_fields->m_showDebug;
            if (m_fields->m_statusLabel) 
                m_fields->m_statusLabel->setVisible(m_fields->m_showDebug);
            if (m_fields->m_debugDrawNode) 
                m_fields->m_debugDrawNode->setVisible(m_fields->m_showDebug);
        }
        PlayLayer::keyDown(key);
    }

    void rl_loop(float dt) {
        if (!m_player1 || !pSharedMem) return;
        
        if (m_fields->m_debugDrawNode) m_fields->m_debugDrawNode->clear();

        // --- HANDLE COMMANDS ---
        if (pSharedMem->reset_command == 1) {
            pSharedMem->reset_command = 0; 
            m_fields->m_stuckFrames = 0;
            this->resetLevel();             
            return;                         
        }

        if (pSharedMem->checkpoint_command == 1) {
            pSharedMem->checkpoint_command = 0;
            if (this->m_isPracticeMode) {
                this->createCheckpoint();
                if (m_fields->m_showDebug && m_fields->m_statusLabel) 
                    m_fields->m_statusLabel->setColor({0, 255, 255});
            }
        } else {
            if (m_fields->m_showDebug && m_fields->m_statusLabel) 
                m_fields->m_statusLabel->setColor({0, 255, 0});
        }

        // --- SPINLOCK: WAIT FOR PYTHON TO FINISH WRITING ---
        // py_writing = 1 means Python is actively writing, so we wait for it to become 0
        int safety = 0;
        while (pSharedMem->py_writing == 1 && safety < 5000) {
            safety++; // Prevent infinite loop
        }

        // LOCK
        pSharedMem->cpp_writing = 1;

        CCPoint pPos = m_player1->getPosition();
        CCRect pRect = m_player1->getObjectRect();

        // --- NEW: CALCULATE REAL VELOCITY ---
        float real_vel_x = 0.0f;
        if (dt > 0.0001f) {
            real_vel_x = (pPos.x - m_fields->m_lastX) / dt;
        }
        m_fields->m_lastX = pPos.x;

        // --- NEW: STUCK / DEATH DETECTION ---
        bool engineDead = m_player1->m_isDead;
        float currentPct = 0.0f;
        if (this->m_levelLength > 0) currentPct = (pPos.x / this->m_levelLength) * 100.0f;

        // If percent hasn't changed for 30 frames (0.5s) and not at start
        if (fabs(currentPct - m_fields->m_lastPercent) < 0.0001f && currentPct > 0.5f && !engineDead) {
            m_fields->m_stuckFrames++;
        } else {
            m_fields->m_stuckFrames = 0;
        }
        m_fields->m_lastPercent = currentPct;

        bool isStuckDead = (m_fields->m_stuckFrames > 30);
        bool effectiveDead = engineDead || isStuckDead;
        bool levelComplete = (currentPct >= 100.0f);

        // --- VEHICLE MODE LOGIC ---
        std::string modeVal = "cube";
        int modeInt = 0;
        if (m_player1->m_isShip) { modeVal = "ship"; modeInt = 1; } 
        
        // --- FILL MEMORY ---
        pSharedMem->player_x = pPos.x;
        pSharedMem->player_y = m_player1->getPositionY();
        pSharedMem->player_vel_x = real_vel_x; // Using real velocity
        pSharedMem->player_vel_y = m_player1->m_yVelocity;
        pSharedMem->player_rot = m_player1->getRotation();
        pSharedMem->gravity = m_player1->m_isUpsideDown ? -1 : 1;
        pSharedMem->is_on_ground = m_player1->m_isOnGround;
        pSharedMem->is_dead = effectiveDead;
        pSharedMem->is_terminal = effectiveDead || levelComplete;
        pSharedMem->percent = currentPct;
        pSharedMem->player_mode = modeInt;
        pSharedMem->player_speed = m_player1->m_playerSpeed;

        // --- SCAN OBJECTS & CALCULATE REWARD DISTANCES ---
        std::vector<RLObject> nearbyObjects;
        float nearestHazard = 9999.0f;
        float nearestSolid = 9999.0f;
        
        CCObject* obj = nullptr;
        CCARRAY_FOREACH(m_objects, obj) {
            auto go = typeinfo_cast<GameObject*>(obj);
            if (!go) continue;

            CCRect objRect = go->getObjectRect();
            float dx = objRect.getMinX() - pRect.getMaxX();
            float dy = objRect.getMidY() - pRect.getMidY();

            if (dx < -50.0f || dx > 800.0f) continue;

            std::string cat = getObjectCategory(go);
            if (cat == "decoration" || cat == "unknown") continue;

            // --- REWARD SHAPING: Find nearest distances ---
            if (dx > 0) { // Only objects in front
                if (cat == "spike" && dx < nearestHazard) nearestHazard = dx;
                if (cat == "solid_block" && dx < nearestSolid) nearestSolid = dx;
            }

            RLObject rlo;
            rlo.category = cat;
            rlo.rect = objRect;
            rlo.dx = dx;
            rlo.dy = dy;
            rlo.w = objRect.size.width;
            rlo.h = objRect.size.height;
            nearbyObjects.push_back(rlo);
        }

        // Store distances in shared memory
        pSharedMem->dist_nearest_hazard = nearestHazard;
        pSharedMem->dist_nearest_solid = nearestSolid;

        std::sort(nearbyObjects.begin(), nearbyObjects.end(), [](const RLObject &a, const RLObject &b){
            return a.dx < b.dx;
        });

        // --- WRITE OBJECTS TO MEMORY (WITH PADDING) ---
        for(int i=0; i<MAX_OBJECTS; i++) {
            if (i < nearbyObjects.size()) {
                pSharedMem->objects[i].dx = nearbyObjects[i].dx;
                pSharedMem->objects[i].dy = nearbyObjects[i].dy;
                pSharedMem->objects[i].w = nearbyObjects[i].w;
                pSharedMem->objects[i].h = nearbyObjects[i].h;
                pSharedMem->objects[i].type = categoryToInt(nearbyObjects[i].category);
            } else {
                // SAFE PADDING
                pSharedMem->objects[i].dx = 9999.0f;
                pSharedMem->objects[i].dy = 0.0f;
                pSharedMem->objects[i].w = 0.0f;
                pSharedMem->objects[i].h = 0.0f;
                pSharedMem->objects[i].type = -1; // Empty
            }
        }

        pSharedMem->cpp_writing = 0; // UNLOCK

        // --- VISUALIZATION (PRESERVED) ---
        if (m_fields->m_showDebug) {
            if (m_fields->m_statusLabel) {
                std::ostringstream vis;
                vis << std::fixed << std::setprecision(2);

                vis << "=== RL SHARED MEMORY ===\n";
                vis << "X Pos: " << std::left << std::setw(8) << pSharedMem->player_x 
                    << " | Grounded: " << (pSharedMem->is_on_ground ? "TRUE" : "FALSE") << "\n";
                
                vis << "Mode: " << std::left << std::setw(7) << modeVal
                    << " Pct: " << pSharedMem->percent << "%\n";

                // Updated to show Real Velocity
                vis << "VelX: " << real_vel_x << " | VelY: " << m_player1->m_yVelocity << "\n";
                vis << "Act : " << (pSharedMem->action_command ? "PRESS (1)" : "IDLE (0)")
                    << " | Reset: " << pSharedMem->reset_command << "\n";
                vis << "Term: " << (pSharedMem->is_terminal ? "YES" : "NO") << "\n\n";

                vis << " ID | Cat             | dX  | dY  | W  | H \n";
                vis << " ---+----------------+-----+-----+----+----\n";

                for(size_t i=0; i<nearbyObjects.size(); ++i) {
                    if (i >= MAX_OBJECTS) break;
                    auto& o = nearbyObjects[i];
                    vis << " " << std::left << std::setw(3) << i
                        << "| " << std::setw(15) << o.category.substr(0, 14)
                        << "| " << std::setw(4) << (int)o.dx
                        << "| " << std::setw(4) << (int)o.dy
                        << "| " << std::setw(3) << (int)o.w
                        << "| " << (int)o.h << "\n";
                }
                m_fields->m_statusLabel->setString(vis.str().c_str());

                if(m_fields->m_debugDrawNode) {
                    // 1. Draw Player Bounding Box
                    CCPoint pPoints[] = {
                        ccp(pRect.getMinX(), pRect.getMinY()), ccp(pRect.getMaxX(), pRect.getMinY()),
                        ccp(pRect.getMaxX(), pRect.getMaxY()), ccp(pRect.getMinX(), pRect.getMaxY())
                    };
                    m_fields->m_debugDrawNode->drawPolygon(pPoints, 4, {0,0,0,0}, 1, {1,1,1,1});

                    // 2. DRAW ALL VISIBLE OBJECTS
                    CCObject* debugObj = nullptr;
                    CCARRAY_FOREACH(m_objects, debugObj) {
                        auto go = typeinfo_cast<GameObject*>(debugObj);
                        if (!go || go->m_objectID == 0) continue;

                        CCRect objRect = go->getObjectRect();
                        if (fabsf(objRect.getMidX() - pPos.x) > 1000.0f) continue;

                        std::string cat = getObjectCategory(go);
                        if (cat == "decoration" || cat == "unknown") continue; 

                        ccColor4F border = {0.5f, 0.5f, 0.5f, 1.0f}; 
                        if (cat == "spike") border = {1.0f, 0.0f, 0.0f, 1.0f};
                        else if (cat == "solid_block") border = {0.0f, 1.0f, 0.0f, 1.0f};
                        else if (cat.find("portal") != std::string::npos) border = {1.0f, 1.0f, 0.0f, 1.0f};

                        CCPoint v[] = {
                            ccp(objRect.getMinX(), objRect.getMinY()), ccp(objRect.getMaxX(), objRect.getMinY()),
                            ccp(objRect.getMaxX(), objRect.getMaxY()), ccp(objRect.getMinX(), objRect.getMaxY())
                        };
                        m_fields->m_debugDrawNode->drawPolygon(v, 4, {0,0,0,0}, 1.0f, border);
                    }

                    // 3. DRAW RL Objects Overlay
                    for (size_t i = 0; i < nearbyObjects.size(); ++i) {
                        if (i >= MAX_OBJECTS) break;
                        auto& o = nearbyObjects[i];
                        ccColor4F rlBorder = {1.0f, 1.0f, 1.0f, 1.0f};
                        CCPoint v[] = {
                            ccp(o.rect.getMinX(), o.rect.getMinY()), ccp(o.rect.getMaxX(), o.rect.getMinY()),
                            ccp(o.rect.getMaxX(), o.rect.getMaxY()), ccp(o.rect.getMinX(), o.rect.getMaxY())
                        };
                        m_fields->m_debugDrawNode->drawPolygon(v, 4, {0,0,0,0}, 2.0f, rlBorder);
                    }
                }
            }
        } // END m_showDebug check

        // --- APPLY ACTION ---
        // The action_command represents the desired button state (0=released, 1=pressed)
        // We need to track transitions to trigger pushButton/releaseButton only when state changes
        int cmd = pSharedMem->action_command;
        
        // DEBUG: Log action reads (first 5 times only)
        static int action_log_count = 0;
        if (action_log_count < 5) {
            log::info("DEBUG: action_command = {}, m_isHolding = {}", cmd, m_fields->m_isHolding);
            action_log_count++;
        }
        
        if (cmd == 1) {
            // Action is PRESS
            if (!m_fields->m_isHolding) {
                // Transition: not holding -> holding. Send push event
                m_player1->pushButton(PlayerButton::Jump);
                m_fields->m_isHolding = true;
            }
            // If already holding, do nothing (button already pressed)
        } else {
            // Action is RELEASE  
            if (m_fields->m_isHolding) {
                // Transition: holding -> not holding. Send release event
                m_player1->releaseButton(PlayerButton::Jump);
                m_fields->m_isHolding = false;
            }
            // If already released, do nothing (button already released)
        }
    }
};