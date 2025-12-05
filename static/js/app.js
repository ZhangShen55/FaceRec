// app.js

// 页面加载完成后立即执行
document.addEventListener('DOMContentLoaded', () => {
    // 1. 加载已存在的人物列表
    loadPersons();

    // 2. 绑定“添加人物”表单的提交事件
    const addPersonForm = document.getElementById('add-person-form');
    addPersonForm.addEventListener('submit', handleAddPerson);

    // 3. 绑定“在线识别”表单的提交事件
    const recognitionForm = document.getElementById('recognition-form');
    recognitionForm.addEventListener('submit', handleRecognizeFace);
    
    // 4. 为识别图片输入框添加预览功能
    const recognitionImageInput = document.getElementById('recognition-image');
    recognitionImageInput.addEventListener('change', (event) => {
        const preview = document.getElementById('image-preview');
        const file = event.target.files[0];
        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';
        }
    });
    // 以下是添加
    const batchBtn = document.getElementById('btn-batch-upload');
    if (batchBtn) batchBtn.addEventListener('click', handleBatchUpload);
    // 绑定查询表单
    const searchForm = document.getElementById('search-person-form');
    searchForm.addEventListener('submit', handleSearchPerson);

    // 绑定删除按钮点击事件
    const deleteBtn = document.getElementById('confirm-delete');
    if (deleteBtn) {
        deleteBtn.addEventListener('click', handleDeletePerson);  // 绑定删除按钮事件
    }
    // 以上是添加

});

// 记录待删除的人物 ID
let personToDelete = null;

// 显示删除确认模态框
function showDeleteModal(personId, personName) {
    personToDelete = personId;  // 保存待删除人物的 ID
    const modal = new bootstrap.Modal(document.getElementById('delete-person-modal'));  // 初始化模态框
    document.getElementById('deletePersonModalLabel').textContent = `确认删除 ${personName}`;
    modal.show();  // 显示模态框
}

// 绑定删除按钮点击事件
async function handleDeletePerson() {
    if (!personToDelete) {
        console.error("No person selected to delete.");
        return;  // 如果没有设置待删除的 ID，则返回
    }
    console.log(`Attempting to delete person with ID: ${personToDelete}`);
    try {
        const response = await fetch(`/persons/delete?person_id=${personToDelete}`, {
            method: 'DELETE',
        });
        const result = await response.json();

        if (!response.ok) {
            alert(result.detail || '删除失败');
            return;
        }

        alert(`成功删除人物: ${result.message}`);
        // 重新查询人物，刷新页面
        await handleSearchPerson(event);  // 异步调用搜索，刷新页面
    } catch (error) {
        console.error('Error deleting person:', error);
    } finally {
        personToDelete = null; // 清空删除人物的 ID
    }
}


// 点击关闭或取消按钮时清理 personToDelete
document.querySelectorAll('[data-bs-dismiss="modal"]').forEach(button => {
    button.addEventListener('click', () => {
        personToDelete = null; // 清空待删除的人物 ID
    });
});

// 简单的工具：更新进度条与统计
function updateBatchUI(done, totalImages, success, fail, skip, totalFiles) {
    const stats = document.getElementById('batch-stats');
    stats.style.display = 'block';
    document.getElementById('batch-total').textContent = totalFiles;
    document.getElementById('batch-images').textContent = totalImages;
    document.getElementById('batch-success').textContent = success;
    document.getElementById('batch-fail').textContent = fail;
    document.getElementById('batch-skip').textContent = skip;
    const pct = totalImages ? Math.round((done / totalImages) * 100) : 0;
    const bar = document.getElementById('batch-progressbar');
    bar.style.width = pct + '%';
    bar.textContent = pct + '%';
}



// 显示提示信息的辅助函数
function showAlert(message, type = 'danger', containerId) {
    const container = document.getElementById(containerId);
    const wrapper = document.createElement('div');
    wrapper.innerHTML = [
        `<div class="alert alert-${type} alert-dismissible" role="alert">`,
        `   <div>${message}</div>`,
        '   <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
        '</div>'
    ].join('');
    container.append(wrapper);
}


// 函数：加载人物列表
async function loadPersons() {
    try {
        const response = await fetch('/persons');
        if (!response.ok) throw new Error('获取人物列表失败');
        
        const persons = await response.json();
        const personListDiv = document.getElementById('person-list');
        personListDiv.innerHTML = ''; // 清空现有列表

        persons.forEach(person => {
            const col = document.createElement('div');
            col.className = 'col';
            col.innerHTML = `
                <div class="card h-100">
                    <img src="${person.photo_path}" class="card-img-top" alt="${person.name}">
                    <div class="card-body">
                        <h6 class="card-title text-center">${person.name}</h6>
                    </div>
                </div>
            `;
            personListDiv.appendChild(col);
        });
    } catch (error) {
        console.error('Error loading persons:', error);
        showAlert(error.message, 'danger', 'alert-container-add');
    }
}

// 函数：处理添加人物的表单提交
async function handleAddPerson(event) {
    event.preventDefault(); // 阻止表单默认的刷新页面行为
    
    const form = event.target;
    const formData = new FormData(form);

    try {
        const response = await fetch('/persons', {
            method: 'POST',
            body: formData,
            // 注意：使用FormData时，浏览器会自动设置正确的Content-Type，无需手动指定
        });

        const result = await response.json();

        if (!response.ok) {
            // 如果后端返回错误信息（如：未检测到人脸）
            throw new Error(result.detail || '添加失败，请检查输入。');
        }
        
        showAlert(`人物 "${result.name}" 添加成功!`, 'success', 'alert-container-add');
        form.reset(); // 清空表单
        loadPersons(); // 重新加载人物列表
    } catch (error) {
        console.error('Error adding person:', error);
        showAlert(error.message, 'danger', 'alert-container-add');
    }
}

// 函数：处理人脸识别的表单提交
async function handleRecognizeFace2(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    const resultCard = document.getElementById('recognition-result-card');
    
    try {
        const response = await fetch('/recognize', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
             throw new Error(result.message || '识别失败');
        }

        // 更新结果显示区域
        document.getElementById('result-name').textContent = result.name;
        document.getElementById('result-similarity').textContent = result.similarity;
        resultCard.style.display = 'block'; // 显示结果卡片

    } catch (error) {
        console.error('Error recognizing face:', error);
        resultCard.style.display = 'none'; // 隐藏结果卡片
        showAlert(error.message, 'danger', 'alert-container-rec');
    }
}

async function handleRecognizeFace(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    const resultCard = document.getElementById('recognition-result-card');
    const detectedCard = document.getElementById('detected-face-card');
    const detectedImg = document.getElementById('detected-face-img');
    const detectedBbox = document.getElementById('detected-bbox');

    try {
        const response = await fetch('/recognize', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            // 例如：未检测到人脸
            detectedCard.style.display = 'none';
            resultCard.style.display = 'none';
            throw new Error(result.message || '识别失败');
        }

        // 显示检测到的人脸裁剪图（无论是否匹配到人物）
        detectedImg.src = result.detected_face_url;
        detectedBbox.textContent = result.bbox
            ? `bbox: x=${result.bbox.x}, y=${result.bbox.y}, w=${result.bbox.w}, h=${result.bbox.h}`
            : '';
        detectedCard.style.display = 'block';

        // 显示识别结果
        const statusEl = document.getElementById('result-status');
        if (result.match && result.match.matched) {
            document.getElementById('result-name').textContent = result.match.name || '';
            document.getElementById('result-similarity').textContent = result.match.similarity || '';
            statusEl.textContent = '识别成功';
        } else {
            document.getElementById('result-name').textContent = '未匹配到该人物';
            document.getElementById('result-similarity').textContent = result.match?.similarity || '--';
            statusEl.textContent = result.message || '未匹配到该人物';
        }
        resultCard.style.display = 'block';

    } catch (error) {
        console.error('Error recognizing face:', error);
        resultCard.style.display = 'none';
        showAlert(error.message, 'danger', 'alert-container-rec');
    }
}


// 批量上传核心逻辑：逐个调用 /persons
async function handleBatchUpload() {
    const input = document.getElementById('folder-input');
    const files = Array.from(input.files || []);
    const alertBoxId = 'alert-container-batch';

    if (!files.length) {
        showAlert('请先选择一个包含图片的文件夹。', 'warning', alertBoxId);
        return;
    }

    // 过滤掉以点（.）开头的隐藏文件
    const imageFiles = files.filter(f => !f.name.startsWith('.'));

    // 分类：图片 vs 非图片
    const isImage = (f) => f.type.startsWith('image/') ||
        /\.(jpg|jpeg|png|bmp|webp)$/i.test(f.name);
    const validFiles = imageFiles.filter(isImage);
    const skipCount = imageFiles.length - validFiles.length;

    let success = 0, fail = 0, done = 0;
    const failedFiles = []; // 记录失败的文件名
    updateBatchUI(done, validFiles.length, success, fail, skipCount, files.length);

    // 顺序上传（最稳），如需更快可做并发队列
    for (const file of validFiles) {
        const nameWithoutExt = file.name.replace(/\.[^.]+$/, '');
        const fd = new FormData();
        fd.append('name', nameWithoutExt);
        // description 不传：后端会是 None
        fd.append('photo', file);

        try {
            const resp = await fetch('/persons', { method: 'POST', body: fd });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || '入库失败');
            success += 1;
        } catch (e) {
            console.error('入库失败:', file.name, e);
            fail += 1;
            failedFiles.push(file.name);  // 记录失败文件
        } finally {
            done += 1;
            updateBatchUI(done, validFiles.length, success, fail, skipCount, files.length);
        }
    }

    // 入库结束
    if (fail === 0) {
        showAlert(`批量入库完成，成功 ${success} / ${validFiles.length}。`, 'success', alertBoxId);
    } else {
        showAlert(`批量入库完成，成功 ${success}，失败 ${fail}，跳过（非图片）${skipCount}。`, 'warning', alertBoxId);
    }

    // 显示失败的文件名
    if (failedFiles.length > 0) {
        showAlert(`失败的文件：<br>${failedFiles.join('<br>')}`, 'danger', alertBoxId);
    }

    // 刷新右侧人物列表
    loadPersons();
}


async function handleSearchPerson(event) {
    event.preventDefault();  // 确保传递 event

    const name = document.getElementById('search-name').value;
    const id = document.getElementById('search-id').value;

    let url = '/persons/search';
    let params = new URLSearchParams();

    if (name) {
        params.append('name', name);
    }

    if (id) {
        params.append('person_id', id);
    }

    url += '?' + params.toString();

    try {
        const response = await fetch(url);
        const result = await response.json();

        if (!response.ok) {
            alert(result.detail || '查询失败');
            return;
        }

        const resultsDiv = document.getElementById('search-results');
        resultsDiv.innerHTML = '';

        result.persons.forEach(person => {
            const personCard = document.createElement('div');
            personCard.classList.add('col');
            personCard.innerHTML = `
                <div class="card">
                    <img src="${person.photo_url}" class="card-img-top" alt="${person.name}">
                    <div class="card-body">
                        <h6 class="card-title">${person.name}</h6>
                        <button class="btn btn-danger" onclick="showDeleteModal('${person.id}', '${person.name}')">删除</button>
                    </div>
                </div>
            `;
            resultsDiv.appendChild(personCard);
        });
    } catch (error) {
        console.error('Error searching persons:', error);
    }
}
