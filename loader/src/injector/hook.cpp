#include <cstring>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/resource.h>
#include <unistd.h>
#include <unwind.h>

#include <lsplt.hpp>

#include "android_util.hpp"
#include "daemon.hpp"
#include "module.hpp"
#include "zygisk.hpp"

using namespace std;

// *********************
// Zygisk Bootstrapping
// *********************
//
// Zygisk's lifecycle is driven by several PLT function hooks in libandroid_runtime, libart, and
// libnative_bridge. As Zygote is starting up, these carefully selected functions will call into
// the respective lifecycle callbacks in Zygisk to drive the progress forward.
//
// The entire bootstrap process is shown in the graph below.
// Arrows represent control flow, and the blocks are sorted chronologically from top to bottom.
//
//       libandroid_runtime                zygisk                 libart
//
//           ┌───────┐                 ┌─────────────┐
//           │ start │                 │ remote_call │
//           └───┬───┘                 └──────┬──────┘
//               │                            │
//               │                            ▼
//               │                        ┌────────┐
//               │                        │hook_plt│
//               │                        └────────┘
//               ▼
//   ┌──────────────────────┐
//   │ strdup("ZygoteInit") │
//   └───────────┬────┬─────┘
//               │    │                ┌───────────────┐
//               │    └───────────────►│hook_zygote_jni│
//               │                     └───────────────┘       ┌─────────┐
//               │                                             │         │
//               └────────────────────────────────────────────►│   JVM   │
//                                                             │         │
//                                                             └──┬─┬────┘
//     ┌───────────────────┐                                      │ │
//     │nativeXXXSpecialize│◄─────────────────────────────────────┘ │
//     └─────────────┬─────┘                                        │
//                   │                 ┌─────────────┐              │
//                   └────────────────►│ZygiskContext│              │
//                                     └─────────────┘              ▼
//                                                       ┌─────────────────────────┐
//                                                       │pthread_attr_setstacksize│
//                                                       └──────────┬──────────────┘
//                                    ┌────────────────┐            │
//                                    │restore_plt_hook│◄───────────┘
//                                    └────────────────┘
//
// Some notes regarding the important functions/symbols during bootstrap:
//
// * HookContext::hook_plt(): hook functions like |unshare| and |strdup|
// * strdup: called in AndroidRuntime::start before calling ZygoteInit#main(...)
// * HookContext::hook_zygote_jni(): replace the process specialization functions registered
//   with register_jni_procs. This marks the final step of the code injection bootstrap process.
// * pthread_attr_setstacksize: called whenever the JVM tries to setup threads for itself. We use
//   this method to cleanup and unmap Zygisk from the process.

constexpr const char *kZygoteInit = "com.android.internal.os.ZygoteInit";
constexpr const char *kZygote = "com/android/internal/os/Zygote";

// Global contexts:
//
// HookContext lives as long as Zygisk is loaded in memory. It tracks the process's function
// hooking state and bootstraps code injection until we replace the process specialization methods.
//
// ZygiskContext lives during the process specialization process. It implements Zygisk
// features, such as loading modules and customizing process fork/specialization.

ZygiskContext *g_ctx;
HookContext *g_hook;

// -----------------------------------------------------------------

#define DCL_HOOK_FUNC(ret, func, ...)                                                              \
    ret (*old_##func)(__VA_ARGS__);                                                                \
    ret new_##func(__VA_ARGS__)

DCL_HOOK_FUNC(static char *, strdup, const char *str) {
    if (strcmp(kZygoteInit, str) == 0) {
        g_hook->hook_zygote_jni();
        g_hook->cached_map_infos = lsplt::MapInfo::Scan();
    }
    return old_strdup(str);
}

// Skip actual fork and return cached result if applicable
DCL_HOOK_FUNC(int, fork) { return (g_ctx && g_ctx->pid >= 0) ? g_ctx->pid : old_fork(); }

// ---------------------------------------------------------------------------
// GrapheneOS exec-spawn support (opt-in, Stage 1: detection/observability only)
// ---------------------------------------------------------------------------
//
// On GrapheneOS "secure app spawning", the primary zygote launches apps with
// Zygote.nativeForkExec(): the child _Fork()s and immediately re-execs
// app_process (execveat on arm64, execve as fallback), replaying the recorded
// zygote command. Because the child execs, it loses the injected Zygisk loader,
// so modules never reach exec-spawned apps.
//
// The exec happens inside the child while it is still a plain fork of the
// injected zygote -- meaning this PLT hook (installed in the zygote, inherited
// across _Fork) runs in the child, right before the exec. The replayed command
// always carries ExecSpawning.COMMAND_FD_ARG ("--command-fd="), which uniquely
// identifies an exec-spawn launch and never matches system_server or regular
// forkAndSpecialize children (those do not exec).
//
// Stage 1 only observes and logs. Stage 2 will, at this point, deterministically
// hand the child off for injection (self-SIGSTOP so it cannot exec until a root
// tracer has seized it with PTRACE_O_TRACEEXEC). Gated by an opt-in flag read
// once at hook_entry, so default builds are unaffected.

static bool g_exec_spawn_enabled = false;

static bool exec_is_app_process(const char *path) {
    if (!path) return false;
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    return strncmp(base, "app_process", 11) == 0;
}

static bool exec_has_command_fd(char *const argv[]) {
    if (!argv) return false;
    for (int i = 0; argv[i] != nullptr && i < 256; i++) {
        if (strncmp(argv[i], "--command-fd=", 13) == 0) return true;
    }
    return false;
}

static bool is_exec_spawn(const char *path, char *const argv[]) {
    return g_exec_spawn_enabled && exec_is_app_process(path) && exec_has_command_fd(argv);
}

DCL_HOOK_FUNC(static int, execve, const char *path, char *const argv[], char *const envp[]) {
    if (is_exec_spawn(path, argv)) {
        LOGI("exec-spawn: detected app launch pid=%d via execve (%s)", getpid(), path);
        // Stage 2: deterministic sync + handoff to the injector happens here.
    }
    return old_execve(path, argv, envp);
}

DCL_HOOK_FUNC(static int, execveat, int dirfd, const char *path, char *const argv[],
              char *const envp[], int flags) {
    if (is_exec_spawn(path, argv)) {
        LOGI("exec-spawn: detected app launch pid=%d via execveat (%s)", getpid(), path);
        // Stage 2: deterministic sync + handoff to the injector happens here.
    }
    return old_execveat(dirfd, path, argv, envp, flags);
}

// Unmount stuffs in the process's private mount namespace
DCL_HOOK_FUNC(static int, unshare, int flags) {
    if (g_ctx && (flags & CLONE_NEWNS) && !(g_ctx->flags & SERVER_FORK_AND_SPECIALIZE)) {
        bool should_unmount = !(g_ctx->info_flags & (PROCESS_IS_MANAGER | PROCESS_GRANTED_ROOT)) &&
                              g_ctx->flags & DO_REVERT_UNMOUNT;
        if (!should_unmount && g_hook->zygote_unmounted) {
            ZygiskContext::update_mount_namespace(zygiskd::MountNamespace::Root);
        }
        bool is_zygote_clean = g_hook->zygote_unmounted && g_hook->zygote_traces.size() == 0;
        if (should_unmount && !is_zygote_clean) {
            ZygiskContext::update_mount_namespace(zygiskd::MountNamespace::Clean);
        }
    }

    int res = old_unshare(flags);
    errno = 0;  // Restore errno back to 0
    return res;
}

DCL_HOOK_FUNC(int, property_get, const char *key, char *value, const char *default_value) {
    if (!g_hook->skip_hooking_unloader) {
        g_hook->hook_unloader();
        g_hook->skip_hooking_unloader = true;
        for (auto it = g_hook->plt_backup.rbegin(); it != g_hook->plt_backup.rend(); ++it) {
            const auto &[dev, inode, sym, old_func] = *it;
            if (*old_func == old_property_get) {
                if (!lsplt::RegisterHook(dev, inode, sym, *old_func, nullptr) ||
                    !lsplt::CommitHook(g_hook->cached_map_infos, true)) {
                    PLOGE("unhook property_get");
                } else {
                    // A reverse_iterator must be converted to a forward iterator.
                    // The `base()` of the *next* iterator gives the correct position.
                    g_hook->plt_backup.erase(std::next(it).base());
                }
                break;
            }
        }
    }
    return old_property_get(key, value, default_value);
}

// We cannot directly call `munmap` to unload ourselves, otherwise when `munmap` returns,
// it will return to our code which has been unmapped, causing segmentation fault.
// Instead, we hook `pthread_attr_setstacksize` which will be called when VM daemon threads start.
DCL_HOOK_FUNC(static int, pthread_attr_setstacksize, void *target, size_t size) {
    int res = old_pthread_attr_setstacksize((pthread_attr_t *) target, size);

    if (g_hook->should_unmap && gettid() == getpid()) {
        // Only perform unloading on the main thread

        g_hook->restore_plt_hook();
        if (g_hook->should_unmap) {
            void *start_addr = g_hook->start_addr;
            size_t block_size = g_hook->block_size;

            if (g_hook->should_spoof_maps) {
                spoof_virtual_maps("zygisk-module", true);
            }

            delete g_hook;
            // Because both `pthread_attr_setstacksize` and `munmap` have the same function
            // signature, we can use `musttail` to let the compiler reuse our stack frame and thus
            // `munmap` will directly return to the caller of `pthread_attr_setstacksize`.
            LOGV("unmap libzygisk.so loaded at %p with size %zu", start_addr, block_size);
            [[clang::musttail]] return munmap(start_addr, block_size);
        }
        delete g_hook;
    }

    return res;
}

#undef DCL_HOOK_FUNC

// -----------------------------------------------------------------
static size_t get_fd_max() {
    rlimit r{32768, 32768};
    getrlimit(RLIMIT_NOFILE, &r);
    return r.rlim_max;
}

ZygiskContext::ZygiskContext(JNIEnv *env, void *args)
    : env(env),
      args{args},
      process(nullptr),
      pid(-1),
      flags(0),
      info_flags(0),
      allowed_fds(get_fd_max()),
      hook_info_lock(PTHREAD_MUTEX_INITIALIZER) {
    g_ctx = this;
}

ZygiskContext::~ZygiskContext() {
    // This global pointer points to a variable on the stack.
    // Set this to nullptr to prevent leaking local variable.
    // This also disables most plt hooked functions.
    g_ctx = nullptr;

    if (!is_child()) return;

    // Strip out all API function pointers
    for (auto &m : modules) {
        m.clearApi();
    }

    // Cleanup
    g_hook->should_unmap = true;
    g_hook->restore_zygote_hook(env);
}

// -----------------------------------------------------------------

HookContext::HookContext(void *start_addr, size_t block_size)
    : start_addr{start_addr}, block_size{block_size} {};

// -----------------------------------------------------------------

inline void *unwind_get_region_start(_Unwind_Context *ctx) {
    auto fp = _Unwind_GetRegionStart(ctx);
#if defined(__arm__)
    // On arm32, we need to check if the pc is in thumb mode,
    // if so, we need to set the lowest bit of fp to 1
    auto pc = _Unwind_GetGR(ctx, 15);  // r15 is pc
    if (pc & 1) {
        // Thumb mode
        fp |= 1;
    }
#endif
    return reinterpret_cast<void *>(fp);
}

// -----------------------------------------------------------------

void HookContext::register_hook(dev_t dev, ino_t inode, const char *symbol, void *new_func,
                                void **old_func) {
    if (!lsplt::RegisterHook(dev, inode, symbol, new_func, old_func)) {
        LOGE("failed to register plt_hook \"%s\"\n", symbol);
        return;
    }
    plt_backup.emplace_back(dev, inode, symbol, old_func);
}

#define PLT_HOOK_REGISTER_SYM(DEV, INODE, SYM, NAME)                                               \
    register_hook(DEV, INODE, SYM, reinterpret_cast<void *>(new_##NAME),                           \
                  reinterpret_cast<void **>(&old_##NAME))

#define PLT_HOOK_REGISTER(DEV, INODE, NAME) PLT_HOOK_REGISTER_SYM(DEV, INODE, #NAME, NAME)

void HookContext::hook_plt() {
    ino_t android_runtime_inode = 0;
    dev_t android_runtime_dev = 0;

    cached_map_infos = lsplt::MapInfo::Scan();
    for (auto &map : cached_map_infos) {
        if (map.path.ends_with("/libandroid_runtime.so")) {
            android_runtime_inode = map.inode;
            android_runtime_dev = map.dev;
        }
    }

    PLT_HOOK_REGISTER(android_runtime_dev, android_runtime_inode, fork);
    PLT_HOOK_REGISTER(android_runtime_dev, android_runtime_inode, unshare);
    PLT_HOOK_REGISTER(android_runtime_dev, android_runtime_inode, strdup);
    PLT_HOOK_REGISTER(android_runtime_dev, android_runtime_inode, property_get);
    if (g_exec_spawn_enabled) {
        // GrapheneOS exec-spawn: catch the child's re-exec of app_process. execveat is the
        // primary path on arm64; execve is the fallback. Only registered when opted in.
        PLT_HOOK_REGISTER(android_runtime_dev, android_runtime_inode, execve);
        PLT_HOOK_REGISTER(android_runtime_dev, android_runtime_inode, execveat);
    }

    if (!lsplt::CommitHook(cached_map_infos)) LOGE("HookContext::hook_plt failed");

    // Remove unhooked methods
    plt_backup.erase(std::remove_if(plt_backup.begin(), plt_backup.end(),
                                    [](auto &t) { return *std::get<3>(t) == nullptr; }),
                     plt_backup.end());
}

void HookContext::hook_unloader() {
    ino_t art_inode = 0;
    dev_t art_dev = 0;

    cached_map_infos = lsplt::MapInfo::Scan();
    for (auto &map : cached_map_infos) {
        if (map.path.ends_with("/libart.so")) {
            art_inode = map.inode;
            art_dev = map.dev;
            break;
        }
    }

    PLT_HOOK_REGISTER(art_dev, art_inode, pthread_attr_setstacksize);
    if (!lsplt::CommitHook(cached_map_infos)) {
        LOGE("HookContext::hook_unloader failed");
    }
}

void HookContext::restore_plt_hook() {
    // Unhook plt_hook
    for (const auto &[dev, inode, sym, old_func] : plt_backup) {
        if (!lsplt::RegisterHook(dev, inode, sym, *old_func, nullptr)) {
            LOGE("failed to register plt_hook [%s]", sym);
            should_unmap = false;
        }
    }
    if (!lsplt::CommitHook(cached_map_infos, true)) {
        LOGE("failed to restore plt_hook");
        should_unmap = false;
    }
}

// -----------------------------------------------------------------

void HookContext::hook_jni_methods(JNIEnv *env, const char *clz, JNIMethods methods) {
    auto clazz = env->FindClass(clz);
    if (clazz == nullptr) {
        env->ExceptionClear();
        for (auto &method : methods) {
            method.fnPtr = nullptr;
        }
        return;
    }

    vector<JNINativeMethod> hooks;
    for (auto &native_method : methods) {
        // It's useful to allow nullptr function pointer for restoring hook
        if (!native_method.fnPtr) continue;

        auto method_id = env->GetMethodID(clazz, native_method.name, native_method.signature);
        bool is_static = false;
        if (method_id == nullptr) {
            env->ExceptionClear();
            method_id = env->GetStaticMethodID(clazz, native_method.name, native_method.signature);
            is_static = true;
        }
        if (method_id == nullptr) {
            env->ExceptionClear();
            native_method.fnPtr = nullptr;
            continue;
        }
        auto method = util::jni::ToReflectedMethod(env, clazz, method_id, is_static);
        auto modifier = util::jni::CallIntMethod(env, method, member_getModifiers);
        if ((modifier & MODIFIER_NATIVE) == 0) {
            native_method.fnPtr = nullptr;
            continue;
        }
        auto artMethod = util::art::ArtMethod::FromReflectedMethod(env, method);
        hooks.push_back(native_method);
        auto original_method = artMethod->GetData();
        LOGV("replaced %s!%s @%p", clz, native_method.name, original_method);
        native_method.fnPtr = original_method;
    }

    if (hooks.empty()) return;
    env->RegisterNatives(clazz, hooks.data(), hooks.size());
}

void HookContext::hook_zygote_jni() {
    auto get_created_java_vms = reinterpret_cast<jint (*)(JavaVM **, jsize, jsize *)>(
        dlsym(RTLD_DEFAULT, "JNI_GetCreatedJavaVMs"));
    if (!get_created_java_vms) {
        for (auto &map : cached_map_infos) {
            if (!map.path.ends_with("/libnativehelper.so")) continue;
            void *h = dlopen(map.path.data(), RTLD_LAZY);
            if (!h) {
                LOGW("cannot dlopen libnativehelper.so: %s", dlerror());
                break;
            }
            get_created_java_vms =
                reinterpret_cast<decltype(get_created_java_vms)>(dlsym(h, "JNI_GetCreatedJavaVMs"));
            dlclose(h);
            break;
        }
        if (!get_created_java_vms) {
            LOGW("JNI_GetCreatedJavaVMs not found");
            return;
        }
    }
    JavaVM *vm = nullptr;
    jsize num = 0;
    jint res = get_created_java_vms(&vm, 1, &num);
    if (res != JNI_OK || vm == nullptr) return;
    JNIEnv *env = nullptr;
    res = vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
    if (res != JNI_OK || env == nullptr) return;

    auto classMember = util::jni::FindClass(env, "java/lang/reflect/Member");
    if (classMember != nullptr)
        member_getModifiers = util::jni::GetMethodID(env, classMember, "getModifiers", "()I");
    auto classModifier = util::jni::FindClass(env, "java/lang/reflect/Modifier");
    if (classModifier != nullptr) {
        auto fieldId = util::jni::GetStaticFieldID(env, classModifier, "NATIVE", "I");
        if (fieldId != nullptr)
            MODIFIER_NATIVE = util::jni::GetStaticIntField(env, classModifier, fieldId);
    }
    if (member_getModifiers == nullptr || MODIFIER_NATIVE == 0) return;
    if (!util::art::ArtMethod::Init(env)) {
        LOGE("failed to init ArtMethod");
        return;
    }
    hook_jni_methods(env, kZygote, zygote_methods);

    // Each Zygote entrypoint has several version variants in the table and only one
    // resolves per device. If none of a method's variants resolves (e.g. a new OS
    // changed its native signature), that entrypoint is left unhooked and injection
    // silently breaks for the affected process class. Fail loudly instead.
    for (const char *name :
         {"nativeForkAndSpecialize", "nativeSpecializeAppProcess", "nativeForkSystemServer"}) {
        bool hooked = false;
        for (const auto &method : zygote_methods) {
            if (method.fnPtr != nullptr && strcmp(method.name, name) == 0) {
                hooked = true;
                break;
            }
        }
        if (!hooked)
            LOGE("no matching signature for %s; injection will not work on this build", name);
    }
}

void HookContext::restore_zygote_hook(JNIEnv *env) {
    hook_jni_methods(env, kZygote, zygote_methods);
}

// -----------------------------------------------------------------

void hook_entry(void *start_addr, size_t block_size) {
    g_hook = new HookContext(start_addr, block_size);

    // GrapheneOS exec-spawn support is opt-in: enabled only when a flag file exists in the
    // daemon work directory. Read once here (in the zygote); the result is inherited by every
    // forked child, so default builds install no exec hooks and behave exactly like stock.
    {
        std::string flag_path = std::string(zygiskd::GetTmpPath()) + "/exec_spawn_inject";
        g_exec_spawn_enabled = access(flag_path.c_str(), F_OK) == 0;
        LOGI("exec-spawn: injection %s", g_exec_spawn_enabled ? "ENABLED (flag present)"
                                                              : "disabled (no flag)");
    }

    g_hook->hook_plt();

    uintptr_t reserved_base = 0;
    size_t reserved_size = 0;
    clean_linker_trace(zygiskd::GetTmpPath().data(), 1, 0, true, &reserved_base, &reserved_size);

    // The ptracer sums only path-matched mappings, missing the anonymous .bss zeromap at
    // the top of the reservation. soinfo::size is the real figure; trust it only when the
    // base agrees, so a mismatch keeps the old behaviour.
    if (reserved_base == reinterpret_cast<uintptr_t>(start_addr) && reserved_size > block_size) {
        LOGV("linker reserved %zu bytes for us, %zu more than the ptracer measured; unmapping "
             "the whole reservation",
             reserved_size, reserved_size - block_size);
        g_hook->block_size = reserved_size;
    } else if (reserved_base != reinterpret_cast<uintptr_t>(start_addr)) {
        LOGW("linker reservation base %p does not match the injected range at %p; keeping the "
             "ptracer's size %zu",
             (void *) reserved_base, start_addr, block_size);
    }
}

void hookJniNativeMethods(JNIEnv *env, const char *clz, JNINativeMethod *methods, int numMethods) {
    g_hook->hook_jni_methods(env, clz, {methods, (size_t) numMethods});
}
