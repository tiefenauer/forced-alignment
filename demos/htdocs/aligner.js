(function ($) {
    $(document).ready(() => $.ajax({url: 'alignment.json', dataType: 'json', success: onAlignmentLoaded}));

    let onAlignmentLoaded = function (content) {
        // bootstrap alignment
        let player = $('#player')
        let target = $('#target')
        let alignments = content.words

        player.attr('src', 'audio.mp3')
        player.on('timeupdate', alignments, selectWord);
        player.on('seeked', alignments, selectWord);
        player.on('timeupdate', alignments, selectWord);
        player.on('seeked', alignments, selectWord);
        player[0].load();

        alignments.forEach(alignment => align(alignment, player[0], target[0]));
    };

    let align = function (alignment, player, target) {
        // align a simple entry from alignments.json: [text::strt, start::float, stop::float] (time in seconds)
        let text = alignment[0];
        let node = createNode(target, text);
        alignment[3] = node
        $(node).click(() => player.currentTime = alignment[1])
    };

    let isTextNodeContaining = function (text) {
        // checks if a given HTML node contains {text}
        return node => {
            let isTextNode = [3, 4].includes(node.nodeType);
            let containsText = node.data.toLowerCase().includes(text.toLowerCase());
            let isAligned = $(node).hasClass('aligned') || $(node.parentElement).hasClass('aligned')
            return isTextNode && containsText && !isAligned
        }
    }

    let createNode = function (target, text) {
        // replaces all occurrences of {text} in target with a <span class='aligned'>{text}</span>
        let textNodes = getTextNodesIn(target);
        let node = textNodes.find(isTextNodeContaining(text))
        if (node) {
            let wordNode = node.splitText(node.data.toLowerCase().indexOf(text.toLowerCase()))
            wordNode.splitText(text.length)
            let highlightedWord = $('<span></span>').addClass('aligned')
            $(wordNode).replaceWith(highlightedWord)
            highlightedWord.append(wordNode)
            return highlightedWord;
        }
    };

    let getTextNodesIn = function (node, includeWhitespaceNodes) {
        // find all text node children in a parent node
        let textNodes = [], nonWhitespaceMatcher = /\S/;
        function getTextNodes(node) {
            if (node.nodeType === 3) {
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
    };

    let selectWord = function (e) {
        // selects a word by setting the classes and focus
        let alignments = e.data
        $('.current').removeClass('current')
        alignments.forEach(alignment => {
            if (player.currentTime >= alignment[1] && player.currentTime <= alignment[2] && alignment[3]) {
                let node = alignment[3];
                $(node).addClass('current')
                node.focus();
            }
        })
    };
})($)