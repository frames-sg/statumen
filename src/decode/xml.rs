use crate::error::WsiError;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;

/// Maximum XML nesting depth allowed during parsing. Prevents stack overflow
/// from crafted deeply nested XML in slide metadata (Leica, Philips, Ventana).
const MAX_XML_DEPTH: u32 = 256;

/// Find the first element with the given tag name and return its text content.
#[cfg(test)]
pub fn parse_element_text(xml: &str, tag: &str) -> Option<String> {
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let mut inside_tag = false;
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) if e.name().as_ref() == tag.as_bytes() => {
                inside_tag = true;
            }
            Ok(Event::Text(e)) if inside_tag => {
                return e.unescape().ok().map(|s| s.into_owned());
            }
            Ok(Event::End(_)) if inside_tag => {
                return None;
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    None
}

/// Find the first element with the given tag name and return the value of the specified attribute.
#[cfg(test)]
pub fn parse_attribute(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) if e.name().as_ref() == tag.as_bytes() => {
                for a in e.attributes().flatten() {
                    if a.key.as_ref() == attr.as_bytes() {
                        return a.unescape_value().ok().map(|s| s.into_owned());
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    None
}

/// A simple tree representation of an XML document.
///
/// # Security note
///
/// `quick_xml` 0.36 does **not** expand internal DTD entity references by
/// default. The `read_event_into` loop only processes `Start`, `Empty`,
/// `Text`, `End`, and `Eof` events — `DocType` events are ignored. Combined
/// with the recursion depth limit in [`parse_node_recursive`], this makes
/// the parser resistant to billion-laughs-style entity expansion attacks.
#[derive(Debug, Clone)]
pub struct XmlNode {
    pub tag: String,
    pub attributes: HashMap<String, String>,
    pub text: Option<String>,
    pub children: Vec<XmlNode>,
}

impl XmlNode {
    /// Find the first direct child with the given tag name.
    pub fn find(&self, tag: &str) -> Option<&XmlNode> {
        self.children.iter().find(|c| c.tag == tag)
    }

    /// Find all direct children with the given tag name.
    pub fn find_all(&self, tag: &str) -> Vec<&XmlNode> {
        self.children.iter().filter(|c| c.tag == tag).collect()
    }

    /// Get the value of an attribute by name.
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attributes.get(name).map(|s| s.as_str())
    }
}

/// Parse an XML string into a tree of `XmlNode`.
pub fn parse_xml(xml: &str) -> Result<XmlNode, WsiError> {
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();

    // Find the root element
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let node = parse_node_recursive(&e, &mut reader, 0)?;
                return Ok(node);
            }
            Ok(Event::Empty(e)) => {
                return make_empty_node(&e);
            }
            Ok(Event::Eof) => {
                return Err(WsiError::Xml("empty document".into()));
            }
            Err(e) => {
                return Err(WsiError::Xml(e.to_string()));
            }
            _ => {}
        }
        buf.clear();
    }
}

fn make_empty_node(e: &quick_xml::events::BytesStart) -> Result<XmlNode, WsiError> {
    let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
    let mut attributes = HashMap::new();
    for attr in e.attributes().flatten() {
        let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
        let val = attr
            .unescape_value()
            .map_err(|err| WsiError::Xml(err.to_string()))?
            .into_owned();
        attributes.insert(key, val);
    }
    Ok(XmlNode {
        tag,
        attributes,
        text: None,
        children: Vec::new(),
    })
}

fn parse_node_recursive(
    start: &quick_xml::events::BytesStart,
    reader: &mut Reader<&[u8]>,
    depth: u32,
) -> Result<XmlNode, WsiError> {
    if depth > MAX_XML_DEPTH {
        return Err(WsiError::Xml(format!(
            "XML nesting depth exceeds maximum of {}",
            MAX_XML_DEPTH
        )));
    }
    let tag = String::from_utf8_lossy(start.name().as_ref()).into_owned();
    let mut attributes = HashMap::new();
    for attr in start.attributes().flatten() {
        let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
        let val = attr
            .unescape_value()
            .map_err(|err| WsiError::Xml(err.to_string()))?
            .into_owned();
        attributes.insert(key, val);
    }
    let mut children = Vec::new();
    let mut text = None;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let child = parse_node_recursive(&e, reader, depth + 1)?;
                children.push(child);
            }
            Ok(Event::Empty(e)) => {
                children.push(make_empty_node(&e)?);
            }
            Ok(Event::Text(e)) => {
                let t = e
                    .unescape()
                    .map_err(|err| WsiError::Xml(err.to_string()))?
                    .into_owned();
                if !t.trim().is_empty() {
                    text = Some(t);
                }
            }
            Ok(Event::End(_)) => break,
            Ok(Event::Eof) => {
                return Err(WsiError::Xml(format!("unexpected EOF in <{}>", tag)));
            }
            Err(e) => {
                return Err(WsiError::Xml(e.to_string()));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(XmlNode {
        tag,
        attributes,
        text,
        children,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_element_text() {
        let xml = "<root><name>Aperio</name></root>";
        assert_eq!(parse_element_text(xml, "name"), Some("Aperio".to_string()));
    }

    #[test]
    fn test_parse_attribute() {
        let xml = r#"<root><image width="1024"/></root>"#;
        assert_eq!(
            parse_attribute(xml, "image", "width"),
            Some("1024".to_string())
        );
    }

    #[test]
    fn missing_element_returns_none() {
        let xml = "<root><name>Test</name></root>";
        assert_eq!(parse_element_text(xml, "missing"), None);
        assert_eq!(parse_attribute(xml, "missing", "attr"), None);
    }

    #[test]
    fn parse_xml_tree() {
        let xml = r#"<root version="1.0">
            <meta key="vendor">Aperio</meta>
            <levels>
                <level id="0" width="4096" height="2048"/>
                <level id="1" width="2048" height="1024"/>
            </levels>
        </root>"#;

        let root = parse_xml(xml).unwrap();
        assert_eq!(root.tag, "root");
        assert_eq!(root.attr("version"), Some("1.0"));

        let meta = root.find("meta").unwrap();
        assert_eq!(meta.attr("key"), Some("vendor"));
        assert_eq!(meta.text.as_deref(), Some("Aperio"));

        let levels = root.find("levels").unwrap();
        let level_nodes = levels.find_all("level");
        assert_eq!(level_nodes.len(), 2);
        assert_eq!(level_nodes[0].attr("id"), Some("0"));
        assert_eq!(level_nodes[0].attr("width"), Some("4096"));
        assert_eq!(level_nodes[1].attr("id"), Some("1"));
        assert_eq!(level_nodes[1].attr("height"), Some("1024"));

        // children should be empty for self-closing tags
        assert!(level_nodes[0].children.is_empty());
    }

    #[test]
    fn deeply_nested_xml_rejected() {
        let depth = MAX_XML_DEPTH + 10;
        let open_tags: String = (0..depth).map(|i| format!("<n{}>", i)).collect();
        let close_tags: String = (0..depth).rev().map(|i| format!("</n{}>", i)).collect();
        let xml = format!("{}{}", open_tags, close_tags);

        let result = parse_xml(&xml);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("nesting depth"),
            "expected depth error, got: {err_msg}"
        );
    }
}
