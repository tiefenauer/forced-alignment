function createNode(target, text) {
    let textNodes = getTextNodesIn(target)
    textNodes = textNodes.filter(node => [3, 4].includes(node.nodeType))
    for (let ix = 0; ix < textNodes.length; ix++) {
        let node = textNodes[ix]
        let i = node.data.toLowerCase().indexOf(text.toLowerCase());
        if (i > -1) {
            let wordNode = node.splitText(i)
            wordNode.splitText(text.length)
            let highlightedWord = wordNode.ownerDocument.createElement('span');
            wordNode.parentNode.replaceChild(highlightedWord, wordNode);
            highlightedWord.setAttribute('class', 'aligned-word');
            highlightedWord.appendChild(wordNode);
            return highlightedWord;
        }
    }
}

function getTextNodesIn(node, includeWhitespaceNodes) {
    let textNodes = [], nonWhitespaceMatcher = /\S/;

    function getTextNodes(node) {
        if (node.nodeType == 3) {
            if (includeWhitespaceNodes || nonWhitespaceMatcher.test(node.nodeValue)) {
                textNodes.push(node);
            }
        } else {
            for (var i = 0, len = node.childNodes.length; i < len; ++i) {
                getTextNodes(node.childNodes[i]);
            }
        }
    }

    getTextNodes(node);
    return textNodes;
}

function align(alignment) {
    let text = alignment[0];
    let node = createNode(target, text);
    if (node != null) {
        alignment[3] = node;
        $(node).on('click', function () {
            let newTime = Math.floor(alignment[1]);
            // console.log(player)
            // console.log(alignment[0], alignment[1], alignment[2], alignment[3]);
            console.log('setting current time to ' + newTime);
            player[0].currentTime = newTime;
        })
    }
}

function selectWord(player, alignments) {
    $('.aligned-word').removeClass('current-word')
    alignments.forEach(alignment => {
        if (player.currentTime >= alignment[1] && player.currentTime <= alignment[2] && alignment[3]) {
            let node = alignment[3];
            $(node).addClass('current-word')
            node.focus();
        }
    })
}

function onAlignmentLoaded(content) {
    let alignments = content.words

    player.attr('src', 'audio.mp3')
    player[0].load();
    player.on('timeupdate', e => selectWord(e.target, alignments));
    player.on('seeked', e => selectWord(e.target, alignments));
    player[0].addEventListener('timeupdate', e => selectWord(e.target, alignments));
    player[0].addEventListener('seeked', e => selectWord(e.target, alignments));

    alignments.forEach(align);
}

let player = null
let target = null

$(document).ready(function () {
    player = $('#player')
    target = $('#target')[0]

    $.ajax({url: 'alignment.json', dataType: 'json', success: onAlignmentLoaded});
})